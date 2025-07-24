'''
Filter the stems used to create MIXes to include only allowed instruments 
which the model has been trained on
'''

import os
import yaml
import librosa
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import random

# Approved instrument list
ALLOWED_INSTRUMENTS = set([
    '', 'flute', 'french horn', 'viola section', 'viola', 'toms', 'synthesizer',
    'gong', 'bamboo flute', 'alto saxophone', 'clarinet', 'gu', 'zhongruan',
    'distorted electric guitar', 'trombone', 'tack piano', 'violin', 'piccolo',
    'fx/processed sound', 'vibraphone', 'double bass', 'trombone section',
    'tenor saxophone', 'darbuka', 'vocalists', 'harmonica', 'clarinet section',
    'bass drum', 'baritone saxophone', 'sampler', 'flute section', 'violin section',
    'oboe', 'french horn section', 'doumbek', 'horn section', 'female singer',
    'cymbal', 'accordion', 'cello section', 'guzheng', 'tuba', 'liuqin',
    'clean electric guitar', 'bassoon', 'glockenspiel', 'auxiliary percussion',
    'lap steel guitar', 'banjo', 'yangqin', 'acoustic guitar', 'piano',
    'brass section', 'timpani', 'trumpet section', 'scratches', 'trumpet', 'erhu',
    'electric piano', 'bass clarinet', 'dizi', 'mandolin', 'harp', 'drum machine',
    'electric bass', 'tabla', 'claps', 'bongo', 'male rapper', 'male singer',
    'shaker', 'drum set', 'cello', 'oud', 'soprano saxophone', 'tambourine',
    'string section'
])

def load_yaml_metadata(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml_metadata(data, output_path):
    with open(output_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

def loudness_normalize_audio(y, sr, target_lufs=-23.0):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y)
    return pyln.normalize.loudness(y, loudness, target_lufs)

def mix_stems(yaml_obj, base_path, output_dir, target_lufs=-23.0):
    stem_dir = os.path.join(base_path, yaml_obj['stem_dir'])
    original_stems = yaml_obj['stems']

    # Filter allowed stems
    valid_stems = {}
    valid_stem_files = []

    for stem_id, stem in original_stems.items():
        instr = (stem.get('instrument') or '').strip().lower()
        if instr in ALLOWED_INSTRUMENTS:
            file_path = os.path.join(stem_dir, stem['filename'])
            if os.path.exists(file_path):
                valid_stems[stem_id] = stem
                valid_stem_files.append(file_path)

    if not valid_stem_files:
        print("No valid stems to mix.")
        return

    # Load first stem
    y, sr = librosa.load(valid_stem_files[0], sr=None, mono=True)
    y = loudness_normalize_audio(y, sr, target_lufs)

    for stem_file in valid_stem_files[1:]:
        y_add, _ = librosa.load(stem_file, sr=sr, mono=True)
        y_add = loudness_normalize_audio(y_add, sr, target_lufs)

        if len(y) < len(y_add):
            y = np.pad(y, (0, len(y_add) - len(y)))
        elif len(y_add) < len(y):
            y_add = np.pad(y_add, (0, len(y) - len(y_add)))
        
        y += y_add

    y /= len(valid_stem_files)

    # Construct paths
    mix_filename = yaml_obj['mix_filename']
    song_name = os.path.splitext(mix_filename)[0].replace("_MIX", "")
    song_output_dir = os.path.join(output_dir, song_name)
    os.makedirs(song_output_dir, exist_ok=True)

    # Save new mix
    output_mix_path = os.path.join(song_output_dir, mix_filename)
    sf.write(output_mix_path, y, sr)
    print(f"Saved mix: {output_mix_path}")

    # Save new trimmed YAML
    new_yaml_obj = yaml_obj.copy()
    new_yaml_obj['stems'] = valid_stems
    output_yaml_path = os.path.join(song_output_dir, f"{song_name}_METADATA.yaml")
    save_yaml_metadata(new_yaml_obj, output_yaml_path)
    print(f"Saved metadata: {output_yaml_path}")

if __name__ == "__main__":
    base_dir = "/scratch/acisse/MedleyDB_Format/Audio"
    output_base = "/scratch/acisse/Modified_MIX"

    all_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    selected_folders = random.sample(all_folders, min(30, len(all_folders)))

    for folder in selected_folders:
        song_path = os.path.join(base_dir, folder)
        yaml_path = os.path.join(song_path, f"{folder}_METADATA.yaml")

        if os.path.exists(yaml_path):
            print(f"Processing: {folder}")
            yaml_obj = load_yaml_metadata(yaml_path)
            mix_stems(yaml_obj, song_path, output_base, target_lufs=-23.0)
        else:
            print(f"Skipping {folder} (no metadata found)")
