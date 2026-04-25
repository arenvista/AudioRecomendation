import argparse
from audioencoder import AudioProcessor 
from audioencoder import Finder
from spectro import *
import sys

def init():
    parser = argparse.ArgumentParser(
        description="Creates spectrogram data and generates embeddings form .mp3 data"
    )

    parser.add_argument(
        "mode",
        choices=["Spectro", "PCA"], 
        help="The mode you want to run."
    )
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # 3. Parse the arguments
    args = parser.parse_args()
    return args


def handler(args):
    if args.mode == "Spectro":
        print("Generating Spectral Data.")
        BatchProceesor()
    elif args.mode == "PCA":
        print("Generating PCA Data.")
        processor = AudioProcessor()
        finder = Finder()
        input_data_filepath = finder.fuzzy_find_dir("./data/", "Select Directory Containing Songs to Process")
        if input_data_filepath is None: raise ValueError("Path is Empty")
        files = finder.get_all_files(input_data_filepath)
        print(f"Processing => {files}")
        processor.visualize_space(files)
    else:
        return

def main():
    args = init()
    handler(args)

if __name__ == "__main__":
    main()
    print("Exiting.")
