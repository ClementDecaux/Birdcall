import os

import numpy as np
import pandas as pd

import librosa
from pydub import AudioSegment

from scipy.signal import spectrogram

import matplotlib.pyplot as plt

from birdcall.params import DATA_FOLDER_RAW, DATA_FOLDER_PROCESSED

from pydub.exceptions import CouldntDecodeError

from birdcall.utils import make_missing_dirs


class BirdAudio:
    def __init__(self, info: pd.Series):

        self.info = info

        try:
            self.segment = AudioSegment.from_file(self.info['audiopath_mp3'], format="mp3").split_to_mono()[0]
        except CouldntDecodeError:
            self.segment = None

        self.array = None

    def load(self):
        return librosa.load(self.info['audiopath_mp3'], sr=None, mono=False)
    
    def get_array(self):
        '''Get raw data as np.array from audio segment'''
        self.array = np.asarray(self.segment.get_array_of_samples())
        return self.array
    
    def convert_to_wav(self, overwrite=False):

        if not overwrite and self.info['audiopath_wav'] is not np.nan:
            return None
        
        audiofile_out = f"{os.path.splitext(self.info['filename'])[0]}.npy"
        audiopath_out = os.path.join(DATA_FOLDER_RAW, 'train_audio_wav', self.info['ebird_code'], audiofile_out)

        make_missing_dirs(audiopath_out)
        
        return AudioSegment.export(self.segment, audiopath_out, format='wav')

    def get_spectrogram(self, overwrite=False):
        '''Compute spectrogram and save it as .npy file. Can load existing spectrogram file if exist and overwrite is False'''

        if not overwrite and self.info['spectrogram_path'] is not np.nan:
            return np.load(self.info['spectrogram_path'], allow_pickle=True)

        # Compute
        if self.array is None:
            self.get_array()
        f, t, Sxx = spectrogram(self.array, fs=self.segment.frame_rate)

        spectrogram_db = np.array(tuple([t, f, 10*np.log(Sxx + 1e-10)]), dtype=object)

        # Save
        spectrogram_filename = f"{os.path.splitext(self.info['filename'])[0]}.npy"
        filepath = os.path.join(DATA_FOLDER_PROCESSED, 'spectrograms', self.info['ebird_code'], spectrogram_filename)

        make_missing_dirs(filepath)
        
        try:
            np.save(filepath, spectrogram_db)
        except Exception as e: 
            print(e)

        return spectrogram_db

    def display(self):
        if self.array is None:
            self.get_array()
        fig, axes = plt.subplots(2, 1, figsize=(15,5))
        axes[0].plot(self.array)
        axes[1].pcolormesh(*self.get_spectrogram(), shading='gouraud', cmap='Greys')