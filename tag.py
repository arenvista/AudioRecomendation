import os
from audioencoder import Finder
import pandas as pd
from tinytag import TinyTag
from tqdm import tqdm
def update_column(fp):
    song_data = get_mp3_metadata(fp)
    if song_data:
        
        filename_to_match = song_data["Filename"]
        
        if 'Filename' in df.columns:
            # Find the index of the row where the filename matches
            match_idx = df.index[df['Filename'] == filename_to_match]
            
            if not match_idx.empty:
                # Loop through the extracted metadata and add it to the dataframe
                for key, value in song_data.items():
                    if key != "Filename": # Skip filename since it's our match key
                        # This safely adds the value to the specific row. 
                        # If the column (e.g., 'Artist') doesn't exist yet, pandas creates it.
                        df.loc[match_idx, key] = value
        print(f"-> Successfully added metadata to the dataframe for {filename_to_match}.\n")



def get_mp3_metadata(file_path):
    """
    Extracts metadata from an MP3 file.
    
    Args:
        file_path (str): The path to the .mp3 file.
        
    Returns:
        dict: A dictionary containing the extracted metadata, 
              or None if an error occurs.
    """
    # Verify the file exists before attempting to read it
    if not os.path.exists(file_path):
        print(f"Error: The file at '{file_path}' was not found.")
        return None

    try:
        # Load the audio file
        audio = TinyTag.get(file_path)
        
        # Extract desired attributes into a dictionary
        # FIXED: Added the proper key: value syntax for the Filename
        metadata = {
            "Filename": os.path.basename(file_path),
            "Title": audio.title,
            "Artist": audio.artist,
            "Album": audio.album,
            "Album Artist": audio.albumartist,
            "Genre": audio.genre,
            "Year": audio.year,
            "Track Number": audio.track,
            "Total Tracks": audio.track_total,
            "Duration (seconds)": round(audio.duration, 2) if audio.duration else None,
            "Bitrate (kbps)": audio.bitrate,
            "File Size (bytes)": audio.filesize
        }
        
        # Clean up the dictionary by removing any completely empty values (optional)
        metadata = {k: v for k, v in metadata.items() if v is not None and v != ""}
        
        return metadata

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

# --- Example Usage ---
if __name__ == "__main__":
    finder = Finder()
    input_data_filepath = finder.fuzzy_find_dir("./data/", "Select Directory Containing Songs to Process")
    if input_data_filepath is None: raise ValueError("Path is Empty")
    files = finder.get_all_files(input_data_filepath)
    
    # Load the PCA data
    csv_path = "/home/sybil/Documents/School/2026-Spring/CMSC-475/Project/AudioRecomendation/output/pca/2026-04-09_09-02-54_pca.csv"
    df = pd.read_csv(csv_path)
    for fp in tqdm(files):
        # print(f"Extracting metadata for: {fp}...")
        song_data = get_mp3_metadata(fp)
        if song_data:
            
            filename_to_match = song_data["Filename"]
            
            if 'Filename' in df.columns:
                # Find the index of the row where the filename matches
                match_idx = df.index[df['Filename'] == filename_to_match]
                
                if not match_idx.empty:
                    # Loop through the extracted metadata and add it to the dataframe
                    for key, value in song_data.items():
                        if key != "Filename": # Skip filename since it's our match key
                            # This safely adds the value to the specific row. 
                            # If the column (e.g., 'Artist') doesn't exist yet, pandas creates it.
                            df.loc[match_idx, key] = value
                    
                    # print(f"-> Successfully added metadata to the dataframe for {filename_to_match}.\n")
                    
                    # Show the updated row in the console to verify
                    # print("--- Updated DataFrame Row ---")
                    # Pandas might truncate the output, but the data is all there
                    # print(df.loc[match_idx]) 
                    
                    # ---------------------------------------------------------
                    # Optional: Save the updated dataframe back to the CSV file
                    # df.to_csv(csv_path, index=False)
                    # print(f"\n-> Saved updated DataFrame back to {csv_path}")
                    # ---------------------------------------------------------
                    
                else:
                    print(f"-> Warning: No match found in CSV for '{filename_to_match}'.\n")
            else:
                print("-> Warning: The column 'Filename' was not found in the CSV.\n")
    cols = ['Filename', 'Full_Path', 'PC1', 'PC2', 'PC3', 'Title', 'Artist', 'Album', 'Genre', 'Track Number', 'Duration (seconds)', 'Bitrate (kbps)', 'File Size (bytes)', 'Year', 'Album Artist']
    coi = ['Artist', 'Album', 'Genre']
    for c in coi:
        nan_count = df[c].isna().sum()
        print(f"Number of NaNs in column {c}: {nan_count}")
        rows_with_nans = df[df[c].isna()]
        print(rows_with_nans)
    df.to_csv(f"{csv_path}.augmented.csv")
