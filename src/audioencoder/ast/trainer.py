import math
from typing import TypedDict

import evaluate
import numpy as np
from datasets import Audio, load_dataset
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Trainer,
    TrainingArguments,
)


class DataEntry(TypedDict):
    audio_path: str
    text: str


class DataLoader:
    def __init__(self, data_dir="data/audio", test_size=0.2, seed=42):
        self.data_dir = data_dir

        self.dataset = load_dataset("audiofolder", data_dir=self.data_dir)

        self.dataset = self.dataset.class_encode_column("label")

        self.dataset = self.dataset["train"].train_test_split(
            test_size=test_size, seed=seed
        )

        self.labels = self.dataset["train"].features["label"].names
        self.label2id = {label: str(i) for i, label in enumerate(self.labels)}
        self.id2label = {str(i): label for i, label in enumerate(self.labels)}
        self.num_labels = len(self.labels)


class ASTModel:
    def __init__(
        self, model_id="MIT/ast-finetuned-audioset-10-10-0.4593", data_dir="data/audio"
    ):
        self.model_id = model_id
        self.data_loader = DataLoader(data_dir=data_dir)
        self.extractor = AutoFeatureExtractor.from_pretrained(self.model_id)

        self.model = AutoModelForAudioClassification.from_pretrained(
            self.model_id,
            num_labels=self.data_loader.num_labels,
            label2id=self.data_loader.label2id,
            id2label=self.data_loader.id2label,
            ignore_mismatched_sizes=True,
        )

        self.dataset = self.data_loader.dataset.cast_column(
            "audio", Audio(sampling_rate=self.extractor.sampling_rate)
        )

        self.encoded_dataset = self.dataset.map(
            self.preprocess_function,
            remove_columns=["audio", "label"],
            batched=True,
        )

    def preprocess_function(self, examples):
        audio_arrays = [audio["array"] for audio in examples["audio"]]

        inputs = self.extractor(
            audio_arrays,
            sampling_rate=self.extractor.sampling_rate,
            padding="max_length",
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )
        inputs["labels"] = examples["label"]
        return inputs


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)


if __name__ == "__main__":
    print("Initalizing Training")
    # Initialize the AST pipeline
    ast_pipeline = ASTModel(
        data_dir="/home/sybil/Documents/School/2026-Spring/AudioRecomendation/data/audio"
    )
    print("Data Loaded")

    # Setup batch logic to calculate steps per epoch
    batch_size = 8
    grad_accum_steps = 4
    num_epochs_to_train = 10000
    save_every_n_epochs = 500

    total_train_samples = len(ast_pipeline.encoded_dataset["train"])
    steps_per_epoch = math.ceil(total_train_samples / (batch_size * grad_accum_steps))
    save_steps_interval = steps_per_epoch * save_every_n_epochs

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
    )

    print("All Setting Configured")
    trainer = Trainer(
        model=ast_pipeline.model,
        args=training_args,
        train_dataset=ast_pipeline.encoded_dataset["train"],
        eval_dataset=ast_pipeline.encoded_dataset["test"],
        processing_class=ast_pipeline.extractor,  # <--- Changed this line!
        compute_metrics=compute_metrics,
    )

    print("Starting Training")
    trainer.train()

    print("Done")
    trainer.save_model("weights/final-ast-custom-model")
