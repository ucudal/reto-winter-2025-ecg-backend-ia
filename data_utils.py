import base64
import io
import os
import json
from typing import List, Dict
from pymongo import MongoClient
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

def load_data_from_mongo(
    mongo_uri: str,
    db_name: str,
    collection_name: str
) -> List[Dict]:
    """Descarga docs de Mongo y decodifica image_base64→np.array."""
    client = MongoClient(mongo_uri)
    coll = client[db_name][collection_name]
    raw = list(coll.find({}, {"_id":1, "image_base64":1, "label":1}))
    out = []
    for doc in raw:
        img = Image.open(io.BytesIO(base64.b64decode(doc["image_base64"])))
        out.append({
            "_id":    doc["_id"],
            "image":  np.array(img.convert("RGB")),
            "label":  int(doc["label"])
        })
    return out

def get_splits_fixed(
    docs: List[Dict],
    test_ids_path: str, # necesitamos tener localmente los test_ids 
    test_size: float,
    val_size: float,
    seed: int
):
    """
    Split fijo train/val/test.  
    - Test IDs se persisten en test_ids_path.  
    - Devuelve train_docs, val_docs, test_docs.
    """
    ids = [d["_id"] for d in docs]
    labels = [d["label"] for d in docs]

    # primero cargo los test_ids
    # si existen, los leo y asigno a test_ids
    if os.path.exists(test_ids_path):
        with open(test_ids_path, "r") as f:
            test_ids = set(json.load(f))
    else: # si no existen, los creo y guardo
        _, test_ids = train_test_split(
            ids, test_size=test_size, random_state=seed, stratify=labels
        )
        with open(test_ids_path, "w") as f:
            json.dump(list(test_ids), f)

    # como quiero que el test sea fijo, filtro por test_ids
    # y luego hago el split de train/val con el resto
    test_docs = []
    train_val_docs = []
    for d in docs:
        if d["_id"] in test_ids:
            test_docs.append(d)
        else:
            train_val_docs.append(d)

    # split train/val
    tv_labels = [d["label"] for d in train_val_docs]
    train_docs, val_docs = train_test_split(
        train_val_docs,
        test_size=val_size/(1 - test_size),
        random_state=seed,
        stratify=tv_labels
    )
    return train_docs, val_docs, test_docs
