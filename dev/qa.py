from pathlib import Path
import os
import faiss
import pickle

cachepath = Path(os.getenv("CACHE_DIR", Path.home() / ".cache" / "chatlocal"))
indexpath = str(cachepath / "scepa.index")
storepath = cachepath / "scepa.pkl"

index = faiss.read_index(indexpath)

with open(str(storepath), "rb") as f:
    store = pickle.load(f)

store.index = index