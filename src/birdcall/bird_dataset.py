import os

import numpy as np
import pandas as pd

from tqdm import tqdm

from birdcall.bird_audio import BirdAudio
from birdcall.params import DATA_FOLDER, DATA_FOLDER_RAW, DATA_FOLDER_PROCESSED


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

        # Transform sampling rate from str to int
        birds_df['sampling_rate'] = birds_df['sampling_rate'].apply(lambda x: int(x[:-5]))
        # Transform channels (mono 1 - stereo 2) from str to int
        birds_df['channels'] = birds_df['channels'].apply(lambda x: int(x[0]))


        # TODO
        # def  get_datetime(row):
        #     row['date'] row['time']
        # birds_df['datetime'] = birds_df.apply(lambda row, axis=1)

        def check_audio_exists(row, extension):
            filename = f"{os.path.splitext(row['filename'])[0]}.{extension}"
            filepath = os.path.join(DATA_FOLDER_RAW, f'train_audio_{extension}', row['ebird_code'], filename)
            return filepath if os.path.exists(filepath) else np.nan

        birds_df['audiopath_mp3'] = birds_df.apply(lambda row: check_audio_exists(row, 'mp3'), axis=1)
        birds_df['audiopath_wav'] = birds_df.apply(lambda row: check_audio_exists(row, 'wav'), axis=1)

        def check_feature_exists(row, feature):
            feature_filename = f"{os.path.splitext(row['filename'])[0]}.npy"
            filepath = os.path.join(DATA_FOLDER_PROCESSED, feature, row['ebird_code'], feature_filename)
            return filepath if os.path.exists(filepath) else np.nan

        birds_df['spectrogram_path'] = birds_df.apply(lambda row: check_feature_exists(row, 'spectrograms'), axis=1)
        birds_df['mfcc_path'] = birds_df.apply(lambda row: check_feature_exists(row, 'mfcc'), axis=1)
        
        return birds_df

    def get_bird_audio(self, id: int):
        info = self.birds_df.iloc[id]

        try:
            return BirdAudio(info)
        except Exception as e:
            return None

    def convert_to_wav(self):
        for i, row in tqdm(self.birds_df.iterrows()):
            if pd.isna(row['audiopath_wav']):
                bird_audio = self.get_bird_audio(i)
                if bird_audio:
                    bird_audio.convert_to_wav()

    def compute_spectrogram(self):
        for i, row in tqdm(self.birds_df.iterrows()):
            if pd.isna(row['spectrogram_path']) or pd.isna(row['audiopath_wav']):
                bird_audio = self.get_bird_audio(i)
                if bird_audio:
                    bird_audio.get_spectrogram()

    def compute_mfcc(self):
        for i, row in tqdm(self.birds_df.iterrows()):
            if pd.isna(row['mfcc_path']) or pd.isna(row['audiopath_wav']):
                bird_audio = self.get_bird_audio(i)
                if bird_audio:
                    bird_audio.get_mfcc()