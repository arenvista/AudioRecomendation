from typing import TYPE_CHECKING, Any, Dict, Optional
from prompt_toolkit import prompt
from prompt_toolkit.completion import FuzzyWordCompleter
from pathlib import Path
from typing import List, Optional, Union
import sys

class Finder:
    def create_out_dir(self, output_dir: Path) -> Path:
        """Creates the output directory for a given identifier."""
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def get_all_dirs(self, directory: Union[str, Path]) -> List[str]:
        """Recursively fetches all subdir paths relative to the given directory."""
        directory_path = Path(directory)
        return [
            str(p.relative_to(directory_path))
            for p in directory_path.rglob("*")
            if p.is_dir()
        ]

    def fuzzy_find_dir(self, directory_path: Union[str, Path] = ".", message:str="") -> Optional[Path]:
        """Prompts the user to fuzzy search for a file in the directory."""
        print(f"{message}")
        files = self.get_all_dirs(directory_path)

        if not files:
            print("No files found in the specified directory.")
            return None

        completer = FuzzyWordCompleter(files)

        try:
            selected_file = prompt(
                "Type to fuzzy search (Tab to complete) > ", completer=completer
            )

            if selected_file in files:
                full_selected_path = Path(directory_path) / selected_file
                print(f"\nSuccess! You selected: {full_selected_path}")
                return full_selected_path
            else:
                print("\nInvalid selection.")
                return None

        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            return None

    def get_all_files(self, directory: Union[str, Path]) -> List[str]:
        """Recursively fetches all subdir paths relative to the given directory."""
        directory_path = Path(directory)
        return [
            str(directory) + "/" + str(p.relative_to(directory_path))
            for p in directory_path.rglob("*")
            if p.is_file()
        ]


