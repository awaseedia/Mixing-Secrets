'''
Filters activation file to include only stems that are present in each MIX
'''

import os
import yaml
import pandas as pd

def load_yaml_stem_ids(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return set(data.get('stems', {}).keys())

def filter_activation_file(activation_path, yaml_path, output_path):
    # Load valid stems from yaml
    valid_stems = load_yaml_stem_ids(yaml_path)

    # Load activation CSV
    df = pd.read_csv(activation_path)

    # Always keep 'time' column + valid stems
    keep_columns = ['time'] + [col for col in df.columns if col in valid_stems]
    filtered_df = df[keep_columns]

    # Save filtered version
    filtered_df.to_csv(output_path, index=False)
    print(f"Saved filtered activation file to: {output_path}")

if __name__ == "__main__":
    modified_base = "/scratch/acisse/MedleyDB_Format/Modified_MIX"
    original_base = "/scratch/acisse/MedleyDB_Format/Audio"

    for track_name in os.listdir(modified_base):
        mod_dir = os.path.join(modified_base, track_name)
        if not os.path.isdir(mod_dir):
            continue

        lab_file = os.path.join(original_base, track_name, f"{track_name}_ACTIVATION_CONF.lab")
        yaml_file = os.path.join(mod_dir, f"{track_name}_METADATA.yaml")
        out_file = os.path.join(mod_dir, f"{track_name}_ACTIVATION_CONF.lab")

        if os.path.exists(lab_file) and os.path.exists(yaml_file):
            filter_activation_file(lab_file, yaml_file, out_file)
        else:
            print(f"Missing files for: {track_name}")
