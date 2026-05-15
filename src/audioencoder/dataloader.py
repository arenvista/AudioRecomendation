import questionary
import sys
import pandas as pd
from pathlib import Path
import subprocess
from typing import TypedDict

class DataEntry(TypedDict):
    audio_path: str
    text: str

class DataLoader():
    def __init__(self):
        self.csv_filepath = self._select_csv()

    def _select_csv():
        DATA_DIRECTORY = "data/mappings/"
        csv_filepaths = sorted(Path(DATA_DIRECTORY).glob("*.csv"))
        csv_choices = {path.name: path for path in csv_filepaths}

        selected_csv_name = questionary.autocomplete(
            "Select a specific csv file for corresponding data (type to filter/search)  :",
            choices=list(csv_choices.keys()),
            match_middle=True,
            validate=lambda value: value in csv_choices or "Please select a csv file from the list.",
        ).ask()

        selected_csv = csv_choices[selected_csv_name] if selected_csv_name else None
        return selected_csv

    def _obtain_manafest(self):
        data_manifest = pd.read_csv(self.csv_filepath)

        path_exists_cache = {}

        def path_exists(path):
            exists = path_exists_cache.get(path)
            if exists is None:
                exists = Path(path).is_file()
                path_exists_cache[path] = exists
            return exists

        paths = data_manifest[["audio_path", "lyric_path"]]
        missing_mask = ~paths.apply(lambda column: column.map(path_exists))
        missing_files = paths.where(missing_mask).stack().tolist()

        if missing_files:
            raise FileNotFoundError("Missing files:\n" + "\n".join(missing_files))

        def read_lyrics(lyric_path):
            lyrics = Path(lyric_path).read_text(encoding="utf-8")
            if len(lyrics) <= 10:
                raise ValueError(f"File {lyric_path} has insufficient lyrics")
            return lyrics

        data_manifest["lyrics"] = data_manifest["lyric_path"].map(read_lyrics)

        return data_manifest[["audio_path", "lyrics"]]
