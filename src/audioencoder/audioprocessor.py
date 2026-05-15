import librosa
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from datetime import datetime 

class AudioProcessor:
    def __init__(self, model_id="MIT/ast-finetuned-audioset-10-10-0.4593"):
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModelForAudioClassification.from_pretrained(model_id)
        self.model.eval()
        self.output_dir: Path = Path("output/pca/")
        now = datetime.now()
        self.time_string = now.strftime("%Y-%m-%d_%H-%M-%S")

    def _prepare_audio(self, file_path):
        audio, sr = librosa.load(file_path, sr=16000)
        return self.extractor(audio, sampling_rate=sr, return_tensors="pt")

    def extract_embeddings(self, file_path):
        """Returns a single vector representing the entire audio file."""
        inputs = self._prepare_audio(file_path)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        return outputs.hidden_states[-1].mean(dim=1).squeeze().numpy()

    def visualize_space(self, file_list, labels=None):
            """
            Processes a list of files, plots them in 3D space, and exports PCA to CSV.
            """

            features = []
            print(f"Extracting Embeddings From: {file_list}")
            for f in tqdm(file_list):
                features.append(self.extract_embeddings(f))
            
            # 1. Change PCA components to 3
            pca = PCA(n_components=3)
            reduced_data = pca.fit_transform(features)

            print("Exporting PCA results to CSV...")
            
            csv_data = {
                'Filename': [f.split('/')[-1] for f in file_list],
                'Full_Path': file_list,
                'PC1': reduced_data[:, 0],
                'PC2': reduced_data[:, 1],
                'PC3': reduced_data[:, 2]
            }
            df = pd.read_csv("/home/sybil/Documents/audiorecomendation/data/pca/2026-05-15_17-25-27_pca.csv")
            
            print(df)
            
                
            # Create DataFrame and save
            df = pd.DataFrame(csv_data)
            csv_filepath = self.output_dir / Path(f"{self.time_string}_pca.csv")
            df.to_csv(csv_filepath, index=False)
            print(f"Saved PCA CSV => {csv_filepath} ")
            # ==========================================

            # 2. Setup 3D figure
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            # 3. Plot with 3 dimensions
            scatter = ax.scatter(
                reduced_data[:, 0], 
                reduced_data[:, 1], 
                reduced_data[:, 2], 
                c=range(len(file_list)) if labels is None else labels,
                cmap='viridis',
                s=50
            )
            
            print(f"Annotating PCA")
            for i in tqdm(range(len(file_list))):
                txt = file_list[i].split('/')[-1]
                # 4. Add text in 3D space
                ax.text(
                    reduced_data[i, 0], 
                    reduced_data[i, 1], 
                    reduced_data[i, 2], 
                    txt, 
                    fontsize=8
                )

            ax.set_title("Audio Feature Space (3D PCA)")
            ax.set_xlabel("PC 1")
            ax.set_ylabel("PC 2")
            ax.set_zlabel("PC 3")
            png_filepath = self.output_dir / Path(f"{self.time_string}_pca.png")
            plt.savefig( self.output_dir / Path(f"{self.time_string}_pca.png"))
            print(f"Saved PCA Img => {png_filepath}")
            plt.close()
