import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import librosa
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from scipy.signal import spectrogram

from birdcall.params import DATA_FOLDER_RAW, DATA_FOLDER_PROCESSED
from birdcall.utils import make_missing_dirs


class BirdAudio:
    def __init__(self, info: pd.Series):

        self.info = info

        if pd.isna(self.info['audiopath_wav']):
            self.convert_to_wav()

        self.data, self.sr = librosa.load(os.path.realpath(self.info['audiopath_wav']), sr=None, mono=True)
    
    def convert_to_wav(self, overwrite=False):

        if not overwrite and not pd.isna(self.info['audiopath_wav']):
            return None
        
        audiofile_out = f"{os.path.splitext(self.info['filename'])[0]}.wav"
        audiopath_out = os.path.join(DATA_FOLDER_RAW, 'train_audio_wav', self.info['ebird_code'], audiofile_out)

        make_missing_dirs(audiopath_out)

        try:
            segment = AudioSegment.from_file(self.info['audiopath_mp3'], format="mp3").split_to_mono()[0]
        except CouldntDecodeError as e:
            return None

        return AudioSegment.export(segment, audiopath_out, format='wav')

    def get_spectrogram(self, overwrite=False):
        '''Compute spectrogram and save it as .npy file. Can load existing spectrogram file if exist and overwrite is False'''

        if not overwrite and pd.isna(self.info['spectrogram_path']):
            return np.load(self.info['spectrogram_path'], allow_pickle=True)

        # Compute
        f, t, Sxx = spectrogram(self.data, fs=self.sr)

        spectrogram_db = np.array(tuple([t, f, 10*np.log(Sxx + 1e-10)]), dtype=object)

        # Save
        spectrogram_filename = f"{os.path.splitext(self.info['filename'])[0]}.npy"
        spectrogram_filepath = os.path.join(DATA_FOLDER_PROCESSED, 'spectrograms', self.info['ebird_code'], spectrogram_filename)

        make_missing_dirs(spectrogram_filepath)
        
        try:
            np.save(spectrogram_filepath, spectrogram_db)
        except Exception as e: 
            print(e)

        return spectrogram_db

    def get_mfcc(self, overwrite=False):
        '''Compute mfcc and save it as .npy file. Can load existing mfcc file if exist and overwrite is False'''

        if not overwrite and not pd.isna(self.info['mfcc_path']):
            return np.load(self.info['mfcc_path'], allow_pickle=True)

        # Compute
        mfcc = librosa.feature.mfcc(y=self.data, sr=self.sr)

        # Save
        mfcc_filename = f"{os.path.splitext(self.info['filename'])[0]}.npy"
        mfcc_filepath = os.path.join(DATA_FOLDER_PROCESSED, 'mfcc', self.info['ebird_code'], mfcc_filename)

        make_missing_dirs(mfcc_filepath)
        
        try:
            np.save(mfcc_filepath, mfcc)
        except Exception as e: 
            print(e)

        return mfcc

    def display(self):
        fig, axes = plt.subplots(3, 1, figsize=(20,15))
        axes[0].plot(self.data)
        axes[1].pcolormesh(*self.get_spectrogram(), shading='gouraud', cmap='inferno')
        axes[2].pcolormesh(self.get_mfcc(), shading='gouraud', cmap='inferno')