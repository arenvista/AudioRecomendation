from datetime import datetime
from pathlib import Path

import pandas as pd

audio_dir = Path("audio/")
lyrics_dir = Path("lyrics/")

audio_files = {path.stem: path for path in audio_dir.iterdir() if path.is_file()}
lyrics_files = {path.stem: path for path in lyrics_dir.glob("*.txt") if path.is_file()}

matched_stems = sorted(audio_files.keys() & lyrics_files.keys())

df = pd.DataFrame(
    (
        {
            "audio_path": str(audio_files[stem]),
            "lyrics_path": str(lyrics_files[stem]),
        }
        for stem in matched_stems
    ),
    columns=["audio_path", "lyrics_path"],
)

unmatched_audio = sorted(
    str(audio_files[stem]) for stem in audio_files.keys() - lyrics_files.keys()
)
unmatched_lyrics = sorted(
    str(lyrics_files[stem]) for stem in lyrics_files.keys() - audio_files.keys()
)

print("Audio files not written to CSV:")
print("\n".join(unmatched_audio) if unmatched_audio else "None")

print("\nLyrics files not written to CSV:")
print("\n".join(unmatched_lyrics) if unmatched_lyrics else "None")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_csv = f"mappings/{timestamp}_mapping.csv"
df.to_csv(output_csv, index=False)

print(f"\nSaved {len(df)} mappings to {output_csv}")
