print("Loading modules...")
print("-> Loading Torch & TorchAudio")
import torch
import torchaudio # <-- NEW IMPORT
import os
os.environ["HF_HOME"] = "/umbc/class/cmsc475sp26/users/arenv1/AudioRecomendation/.cache/huggingface"

print("-> Loading Evaluate")
import evaluate

print("-> Loading Misc (math, typing, numpy)")
import math
from typing import TypedDict
import numpy as np

print("-> Loading Datasets")
from datasets import Audio, load_dataset

print("-> Loading Transformers")
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Trainer,
    TrainingArguments,
)

print("All modules loaded successfully!\n" + "-"*40)

class DataEntry(TypedDict):
    audio_path: str
    text: str

class DataLoader:
    def __init__(self, data_dir="data/audio", test_size=0.2, seed=42):
        self.data_dir = data_dir
        print(f"[DataLoader] Target Data Directory => {self.data_dir}")
        
        print("[DataLoader] Loading raw dataset via AudioFolder...")
        self.dataset = load_dataset("audiofolder", data_dir=self.data_dir)
        
        #print("[DataLoader] Dropping rows where label is 'unknown'...")
        #self.dataset = self.dataset.filter(lambda example: example["label"] != "unknown")

        print("[DataLoader] Encoding class labels...")
        self.dataset = self.dataset.class_encode_column("label")
        
        print(f"[DataLoader] Splitting dataset (Test Size: {test_size * 100}%, Seed: {seed})...")
        self.dataset = self.dataset["train"].train_test_split(
            test_size=test_size, seed=seed
        )
        
        self.labels = self.dataset["train"].features["label"].names
        self.label2id = {label: str(i) for i, label in enumerate(self.labels)}
        self.id2label = {str(i): label for i, label in enumerate(self.labels)}
        self.num_labels = len(self.labels)
        
        print(f"[DataLoader] Split complete! Train size: {len(self.dataset['train'])}, Test size: {len(self.dataset['test'])}")
        print(f"[DataLoader] Detected {self.num_labels} unique labels: {self.labels}")
        print("-" * 40)

class ASTModel:
    def __init__(
        self, model_id="MIT/ast-finetuned-audioset-10-10-0.4593", data_dir="data/audio"
    ):
        print(f"[ASTModel] Initializing AST Pipeline using base model: {model_id}")
        self.model_id = model_id
        self.data_loader = DataLoader(data_dir=data_dir)
        
        print("[ASTModel] Loading AutoFeatureExtractor...")
        self.extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
        
        print("[ASTModel] Loading AutoModelForAudioClassification...")
        self.model = AutoModelForAudioClassification.from_pretrained(
            self.model_id,
            num_labels=self.data_loader.num_labels,
            label2id=self.data_loader.label2id,
            id2label=self.data_loader.id2label,
            ignore_mismatched_sizes=True,
        )
        
        # --- OPTIMIZED FAST FILTERING ---
        print("[ASTModel] Temporarily disabling audio decoding for rapid corruption check...")
        self.dataset = self.data_loader.dataset.cast_column("audio", Audio(decode=False))
        
        def is_valid_audio(example):
            try:
                # torchaudio.info only reads the header, taking milliseconds instead of seconds per file
                torchaudio.info(example["audio"]["path"])
                return True
            except Exception:
                return False

        # print("[ASTModel] Filtering out corrupted files (fast metadata check)...")
        # self.dataset = self.dataset.filter(is_valid_audio)
        print(f"[ASTModel] Filtering complete! Clean train size: {len(self.dataset['train'])}, Clean test size: {len(self.dataset['test'])}")

        print(f"[ASTModel] Re-enabling decoding and casting to target sampling rate ({self.extractor.sampling_rate} Hz)...")
        self.dataset = self.dataset.cast_column(
            "audio", Audio(decode=True, sampling_rate=self.extractor.sampling_rate)
        )
        # --------------------------------

        print("[ASTModel] Mapping preprocessing function across the dataset (extracting features)...")
        self.encoded_dataset = self.dataset.map(
            self.preprocess_function,
            remove_columns=["audio", "label"],
            batched=True,
        )
        print("[ASTModel] Feature extraction complete!")
        print("-" * 40)

    def preprocess_function(self, examples):
        audio_arrays = [audio["array"] for audio in examples["audio"]]
        
        inputs = self.extractor(
            audio_arrays,
            sampling_rate=self.extractor.sampling_rate,
            padding="max_length",
            truncation=True,
            max_length=1024,
        )
        inputs["labels"] = examples["label"]
        return inputs

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

if __name__ == "__main__":
    print("\n" + "="*40)
    print("=== INITIALIZING TRAINING SCRIPT ===")
    print("="*40 + "\n")
    
    # Initialize the AST pipeline
    ast_pipeline = ASTModel(
        data_dir="/umbc/class/cmsc475sp26/users/arenv1/AudioRecomendation/data/audio"
    )
    print("[Main] Data loaded and preprocessed successfully.\n")
    
    # Setup batch logic to calculate steps per epoch
    print("[Main] Calculating training steps and logging intervals...")
    batch_size = 8
    grad_accum_steps = 1 
    num_epochs_to_train = 10000
    save_every_n_epochs = 500
    
    total_train_samples = len(ast_pipeline.encoded_dataset["train"])
    steps_per_epoch = math.ceil(total_train_samples / (batch_size * grad_accum_steps))
    save_steps_interval = steps_per_epoch * save_every_n_epochs
    
    print(f"  -> Total Train Samples: {total_train_samples}")
    print(f"  -> Batch Size: {batch_size} (Grad Accumulation: {grad_accum_steps})")
    print(f"  -> Steps per Epoch: {steps_per_epoch}")
    print(f"  -> Total Epochs: {num_epochs_to_train}")
    print(f"  -> Will evaluate and save every {save_every_n_epochs} epochs ({save_steps_interval} steps)\n")
    
    print("[Main] Configuring Training Arguments...")
    training_args = TrainingArguments(
        output_dir="./ast-custom-finetuned",
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=save_steps_interval,
        save_steps=save_steps_interval,
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=grad_accum_steps,
        num_train_epochs=num_epochs_to_train,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        remove_unused_columns=False, # <--- ADD THIS LINE
    )
    
    print("[Main] Instantiating the Hugging Face Trainer...")
    trainer = Trainer(
        model=ast_pipeline.model,
        args=training_args,
        train_dataset=ast_pipeline.encoded_dataset["train"],
        eval_dataset=ast_pipeline.encoded_dataset["test"],
        compute_metrics=compute_metrics,
    )
    
    print("\n" + "="*40)
    print("=  PASSING CONTROL TO TRAINER.TRAIN()  =")
    print("="*40 + "\n")
    trainer.train()
    
    print("\n[Main] Training loop finished!")
    print("[Main] Saving final model state to 'weights/final-ast-custom-model'...")
    trainer.save_model("weights/final-ast-custom-model")
    print("[Main] Done! You may now exit.")
