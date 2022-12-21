import pathlib
import numpy as np
import pydub

from scipy.io.wavfile import write


def make_missing_dirs(path: str):
    '''Make missing folder for a given path'''

    try:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    except Exception as e: 
        print(e)

def quick_audio_save(path: str, data, sr: int, seconds: int):
    write(path, sr, data.astype(np.float32)[:sr*seconds])




