import os
import pickle

def save_trackers(trackers, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(trackers, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_trackers(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)