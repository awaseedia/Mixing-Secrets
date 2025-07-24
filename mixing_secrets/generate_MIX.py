import os
import yaml
import librosa
import numpy as np
import soundfile as sf
import pyloudnorm as pyln

def load_yaml_metadata(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def loudness_normalize_audio(y, sr, target_lufs=-23.0):
    meter = pyln.Meter(sr) 
    loudness = meter.integrated_loudness(y)
    return pyln.normalize.loudness(y, loudness, target_lufs)

def mix_stems(yaml_obj, base_path, target_lufs=-23.0):
    stem_dir = os.path.join(base_path, yaml_obj['stem_dir'])
    mix_path = os.path.join(base_path, yaml_obj['mix_filename'])

    stem_files = [os.path.join(stem_dir, stem['filename']) for stem in yaml_obj['stems'].values()]
    
    if not stem_files:
        print("No stems to mix.")
        return

    # Load first file
    y, sr = librosa.load(stem_files[0], sr=None, mono=True)
    y = loudness_normalize_audio(y, sr, target_lufs)

    # Mix with normalized stems
    for stem_file in stem_files[1:]:
        y_add, _ = librosa.load(stem_file, sr=sr, mono=True)
        y_add = loudness_normalize_audio(y_add, sr, target_lufs)

        if len(y) < len(y_add):
            y = np.pad(y, (0, len(y_add) - len(y)))
        elif len(y_add) < len(y):
            y_add = np.pad(y_add, (0, len(y) - len(y_add)))
        
        y += y_add

    # Normalize final mix to avoid clipping
    y /= len(stem_files)

    # Save mix
    sf.write(mix_path, y, sr)
    print(f"Saved mix: {mix_path}")

if __name__ == "__main__":
    base_dir = "/scratch/acisse/MedleyDB_Format/Audio"
    
    for folder in os.listdir(base_dir):
        song_path = os.path.join(base_dir, folder)
        yaml_path = os.path.join(song_path, f"{folder}_METADATA.yaml")
        
        if os.path.exists(yaml_path):
            print(f"Processing: {folder}")
            yaml_obj = load_yaml_metadata(yaml_path)
            mix_stems(yaml_obj, song_path, target_lufs=-23.0)
        else:
            print(f"Skipping {folder} (no metadata found)")




