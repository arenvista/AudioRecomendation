from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional
import librosa
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import hashlib

from audioencoder.finder import Finder

class SongData:
    def __init__ (self, filepath: Path, output_dir: Path, sr: float|None = None):
        self.filepath: Path = filepath
        print(f"Processing {filepath}")
        self.unique_id: str = hashlib.sha256(str(self.filepath).encode()).hexdigest()
        self.y_audio, self.sample_rate = librosa.load(filepath, sr=sr)
        self.spectrogram_df: pd.DataFrame|None = self.spectrogram_to_dataframe(self.y_audio, int(self.sample_rate))

        self.output_dir: Path = output_dir

        finder = Finder()
        self.output_dir_csv: Path = output_dir.joinpath("csv/")  
        finder.create_out_dir(self.output_dir_csv)
        self.output_path_csv = self.output_dir_csv.joinpath(str(self.filepath.name)+".csv")

        self.output_dir_img: Path = output_dir.joinpath("img/")  
        finder.create_out_dir(self.output_dir_img)
        self.output_path_img = self.output_dir_img.joinpath(str(self.filepath.name)+".png")

    def spectrogram_to_csv(self):
        print(f"\n\t Saving {self.filepath} => {self.output_path_csv}")
        if self.spectrogram_df is not None: self.spectrogram_df.to_csv(f"{self.output_path_csv}")
        else: raise ValueError("No Spectrogram to Save!")

    def spectrogram_to_dataframe(self, y: np.ndarray, sr: int) -> pd.DataFrame:
        """Converts an audio signal's spectrogram into a wide-format pandas DataFrame."""
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # Calculate exact frequencies (Hz) for rows and timestamps (s) for columns
        frequencies = librosa.fft_frequencies(sr=sr)
        times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr)

        # Create the DataFrame
        df_spectrogram = pd.DataFrame(S_db, index=frequencies, columns=times)
        df_spectrogram.index.name = 'Frequency (Hz)'
        
        """Melts a wide-format spectrogram DataFrame into a long (flat) format."""
        df_flat = df_spectrogram.reset_index()
        
        df_flat = df_flat.melt(
            id_vars=['Frequency (Hz)'], 
            var_name='Time (s)', 
            value_name='Amplitude (dB)'
        )

        return df_flat

    def create_spectrogram(self) -> None:
        """Computes the STFT of an audio signal and plots its spectrogram."""
        D = librosa.stft(self.y_audio)

        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        plt.figure(figsize=(12, 6))
        librosa.display.specshow(S_db, sr=self.sample_rate, x_axis='time', y_axis='log', cmap='magma')

        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram (Log Frequency)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()

        if self.output_path_img:
            plt.savefig(self.output_path_img)
            print(f"Spectrogram saved => {self.output_path_img}")
        else: plt.show()

        plt.close()
