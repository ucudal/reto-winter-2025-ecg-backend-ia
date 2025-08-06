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
    collection_name: str,
    output_dir: str = "data/images"
) -> List[Dict]:
    """Descarga docs de Mongo y guarda imágenes en archivos."""
    client = MongoClient(mongo_uri)
    print("1")
    coll = client[db_name][collection_name]
    print("2")
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    print("3")
    
    out = []
    # Usar cursor para procesar documentos uno por uno sin cargar todo en memoria
    cursor = coll.find({}, {"_id":1, "image_base64":1, "label":1})
    print("4")
    
    for doc in cursor:
        print(f"Processing doc with _id: {doc['_id']}")
        try:
            # Decodificar y procesar imagen individualmente
            img = Image.open(io.BytesIO(base64.b64decode(doc["image_base64"])))
            
            # Guardar imagen en archivo
            img_path = os.path.join(output_dir, f"{doc['_id']}.jpg")
            img.convert("RGB").save(img_path)
            
            out.append({
                "_id":    doc["_id"],
                "image_path": img_path,
                "label":  int(doc.get("label", -1)) 
            })
            
            # Liberar memoria explícitamente
            img.close()
            del img
            
        except Exception as e:
            print(f"Error processing doc {doc['_id']}: {e}")
            continue
    
    client.close()
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
    ids = [str(d["_id"]) for d in docs]
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
