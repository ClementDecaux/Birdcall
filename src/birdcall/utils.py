import pathlib

def make_missing_dirs(path: str):
    '''Make missing folder for a given path'''

    try:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    except Exception as e: 
        print(e)