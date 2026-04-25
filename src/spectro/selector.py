from prompt_toolkit import prompt
from prompt_toolkit.completion import FuzzyWordCompleter
from pathlib import Path
from typing import List, Optional, Union
from spectro.spectrogram import SongData
from audioencoder.finder import Finder
import sys

class BatchProceesor:
    def __init__ (self, parent_dir: Path = Path("./data"), output_dir: Path = Path("output/spectro")):
        finder = Finder()
        self.batch_dir = finder.fuzzy_find_dir(parent_dir, "Select Directory Containing Songs to Process")
        if self.batch_dir == None: raise ValueError("Dir Not Found")
        self.song_filepaths = [Path(fp) for fp in finder.get_all_files(self.batch_dir) if fp.endswith(".mp3")]
        finder.create_out_dir(output_dir)
        self.output_dir = output_dir
        self.song_spectrograms = [SongData(fp,self.output_dir) for fp in self.song_filepaths]
        num_songs = len(self.song_spectrograms)
        total_itter = 0
        for i in range(num_songs):
            # Terminal Animation
            total_itter +=1
            percent = 100 * (i + 1) / num_songs
            bar = '█' * int(percent / 5) + '-' * (20 - int(percent / 5))
            move_up = "\033[3F" if total_itter > 1 else "\r" 
            sys.stdout.write(
                f"Songs Processed: |{bar}| {percent:.1f}%\n"
            )
            self.song_spectrograms[i].create_spectrogram()
            # self.song_spectrograms[i].spectrogram_to_csv()

        print(f"Finished Proessing All {num_songs} Tracks")
