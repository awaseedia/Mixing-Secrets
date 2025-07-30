"""
LocalMultiTrack - Mixing Secrets Compatibility Helper

This class provides a lightweight utility for parsing and extracting relevant information
from the Mixing Secrets dataset in a format compatible with the MedleyDB-trained
instrument recognition model used in the MuseDetect framework.

The script reads metadata from each track’s YAML file, extracts instrument labels
assigned to each stem, loads corresponding activation confidence annotations, and
validates the presence of the mix and stem audio files.

The functionality closely mirrors MedleyDB’s MultiTrack structure, allowing seamless
integration of Mixing Secrets tracks into the MuseDetect testing pipeline. This was
developed to enable evaluating the MedleyDB-trained model directly on the Mixing Secrets
dataset without altering the model’s architecture or preprocessing steps.
"""

import yaml
import pandas as pd
from pathlib import Path
import soundfile as sf


class LocalMultiTrack:
    def __init__(self, track_path):
        self.track_path = Path(track_path)
        self.track_name = self.track_path.name

        self.metadata_path = self.track_path / f"{self.track_name}_METADATA.yaml"
        self.activation_path = self.track_path / f"{self.track_name}_ACTIVATION_CONF.lab"

        self._load_metadata()
        self._load_activations()

        self.audio_paths = sorted(self.track_path.glob(f"{self.track_name}_STEM_*.wav"))
        self.has_bleed = False  # Adjust if needed
        self.instruments = self._parse_instruments()

        self.mix_filename = f"{self.track_name}_MIX.wav"
        self.mix_path = self.track_path / self.mix_filename
        
        if not self.mix_path.exists():
            raise FileNotFoundError(f"Missing MIX file: {self.mix_path}")



    def _load_metadata(self):
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        with open(self.metadata_path, "r") as f:
            self.metadata = yaml.safe_load(f)

    def _parse_instruments(self):
        instruments = []
        stems = self.metadata.get("stems", {})
        for stem_id, stem_info in stems.items():
            instrument_info = stem_info.get("instrument")
            if isinstance(instrument_info, dict):
                name = instrument_info.get("name")  # or "family" if you want broader categories
                if name:
                    instruments.append(name)
            elif isinstance(instrument_info, str):
                instruments.append(instrument_info)
        return instruments


    def _load_activations(self):
        if not self.activation_path.exists():
            raise FileNotFoundError(f"Activation file not found: {self.activation_path}")
        
        # Read CSV with comma delimiter and header
        self.activations = pd.read_csv(self.activation_path)
    
        # Ensure 'time' column exists
        if "time" not in self.activations.columns:
            raise ValueError(f"'time' column not found in activation file: {self.activation_path}")


    def get_audio(self, stem_id=None):
        if stem_id is None:
            return [sf.read(p)[0] for p in self.audio_paths]
        else:
            match = [p for p in self.audio_paths if f"STEM_{int(stem_id):02d}" in p.name]
            if not match:
                raise ValueError(f"Stem {stem_id} not found.")
            return sf.read(match[0])[0]

