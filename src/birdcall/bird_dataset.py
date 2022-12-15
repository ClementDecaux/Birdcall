import os

import numpy as np
import pandas as pd

from birdcall.bird_audio import BirdAudio
from birdcall.params import DATA_FOLDER, DATA_FOLDER_RAW, DATA_FOLDER_PROCESSED

from tqdm import tqdm

class BirdAudioDataset:

    def __init__(self, reset=False):
        '''BirdAudioDataset constructor'''
        self.filepath = os.path.join(DATA_FOLDER, 'birds_audios.csv')
        if os.path.exists(self.filepath) and not reset:
            self.birds_df = pd.read_csv(self.filepath, index_col=0)
        else:
            self.birds_df = self.preprocess_train_csv()
            self.birds_df.to_csv(self.filepath)

    def preprocess_train_csv(self):
        '''Load raw train.csv, select features and add audio filepath feature'''

        train_csv = os.path.join(DATA_FOLDER, 'train.csv')
        birds_df = pd.read_csv(train_csv)
        features = ['ebird_code', 'duration', 'sampling_rate', 'channels', 'rating', 'longitude', 'latitude', 'date', 'time', 'url', 'filename']
        birds_df = birds_df[features]

        def check_audio_exists(row, extension):
            filename = f"{os.path.splitext(row['filename'])[0]}.{extension}"
            filepath = os.path.join(DATA_FOLDER_RAW, f'train_audio_{extension}', row['ebird_code'], filename)
            return filepath if os.path.exists(filepath) else np.nan

        birds_df['audiopath_mp3'] = birds_df.apply(lambda row: check_audio_exists(row, 'mp3'), axis=1)
        birds_df['audiopath_wav'] = birds_df.apply(lambda row: check_audio_exists(row, 'wav'), axis=1)

        def check_spectrogram_exists(row):
            spectrogram_filename = f"{os.path.splitext(row['filename'])[0]}.npy"
            filepath = os.path.join(DATA_FOLDER_PROCESSED, 'spectrograms', row['ebird_code'], spectrogram_filename)
            return filepath if os.path.exists(filepath) else np.nan

        birds_df['spectrogram_path'] = birds_df.apply(check_spectrogram_exists, axis=1)
        
        return birds_df

    def get_bird_audio(self, id: int):
        info = self.birds_df.iloc[id]
        return BirdAudio(info)

    def convert_to_wav(self):
        for i, row in tqdm(self.birds_df.iterrows()):
            if row['audiopath_wav'] is np.nan:
                bird_audio = self.get_bird_audio(i)
                if bird_audio.segment:
                    bird_audio.convert_to_wav()

    def compute_spectrogram(self):
        for i, row in tqdm(self.birds_df.iterrows()):
            if row['spectrogram_path'] is np.nan:
                bird_audio = self.get_bird_audio(i)
                if bird_audio.segment:
                    bird_audio.get_spectrogram()