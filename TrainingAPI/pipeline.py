# BY CHATGPT 

import os
from dotenv import load_dotenv
load_dotenv()
import numpy as np
from PIL import Image
import albumentations as A
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

from data_utils   import load_data_from_mongo, get_splits_fixed
from train_model  import train_ecg_model
# from eval_model   import evaluate_model
from onnx_utils import convert_to_onnx, evaluate_onnx_model
from hf_utils import build_test_dataset
from minio import Minio
from datetime import datetime

def get_minio_client():
    return Minio(
        str(os.getenv("MINIO_ENDPOINT")),
        access_key=str(os.getenv("MINIO_ACCESS_KEY")),
        secret_key=str(os.getenv("MINIO_SECRET_KEY")),
        secure=True
    )

def descargar_modelo_actual_minio(dest_path):
    client = get_minio_client()
    bucket = str(os.getenv("MINIO_BUCKET"))
    objeto = str(os.getenv("MINIO_MODEL_OBJECT"))

    try:
        client.fget_object(bucket, objeto, dest_path)
        print(f"✅ Modelo actual descargado desde MinIO a {dest_path}")
        return True
    except Exception as e:
        print(f"⚠️ No se pudo descargar el modelo actual: {e}")
        return False

def subir_nuevo_modelo_a_minio(source_path):
    client = get_minio_client()
    bucket = str(os.getenv("MINIO_BUCKET"))
    objeto = str(os.getenv("MINIO_MODEL_OBJECT"))

    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

    client.fput_object(bucket, objeto, source_path)
    print(f"✅ Nuevo modelo subido a MinIO como {objeto}")

def main():
    # ——— Parámetros ———
    # 1) Cargar desde environment variables
    MONGO_URI         = "mongodb://root:hK43CrNUq1@mongodb.reto-ucu.net:50005/?authSource=admin"
    DB_NAME           = "admin"
    COLL_NAME         = "ecg_collection"
    TEST_IDS_PATH     = "test_ids.json"
    MODEL_OUT_DIR_NEW = "./models/new" 
    MODEL_OUT_DIR_CUR = "./models/current"
    TEST_SIZE, VAL_SIZE, SEED = 0.2, 0.1, 42

    # 1) Carga datos
    print("🔄 Descargando datos de MongoDB...")
    docs = load_data_from_mongo(MONGO_URI, DB_NAME, COLL_NAME, "data/images")
    
    # 2) Split fijo train/val/test
    train_docs, val_docs, test_docs = get_splits_fixed(
        docs, test_ids_path=TEST_IDS_PATH, test_size=TEST_SIZE, val_size=VAL_SIZE, seed=SEED
    )

    print("✅ Datos cargados y divididos:")
    print(f"   - Train: {len(train_docs)} muestras")
    print(f"   - Val: {len(val_docs)} muestras")
    print(f"   - Test: {len(test_docs)} muestras")

    # 3) Entrenamiento
    new_model_pytorch = train_ecg_model(
        train_docs, val_docs,
        model_out_dir=MODEL_OUT_DIR_NEW,
        num_epochs=10,
        batch_size=32,
        learning_rate=2e-4
    )
    print("✅ Nuevo modelo entrenado.")

    # 4) CONVERSIÓN A ONNX
    onnx_path_new = convert_to_onnx(new_model_pytorch, MODEL_OUT_DIR_NEW)
    print(f"✅ Nuevo modelo convertido a ONNX en: {onnx_path_new}")
    
    # 5) Preparar extractor y val_tf
    extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    val_tf    = A.Compose([])
    print("✅ Extractor y transformaciones preparados.")

    # 6) Test dataset
    test_ds = build_test_dataset(test_docs, extractor, val_tf)
    print("✅ Dataset de test preparado.")

    # 7) Evaluar NUEVO modelo (en formato ONNX)
    new_metrics = evaluate_onnx_model(onnx_path_new, test_ds)
    print("✅ Métricas nuevo modelo (ONNX):", new_metrics)

    # 7) Evaluar modelo actual --> hay que trar los datos del bucket 
    # Descargar modelo actual (si existe)
    onnx_path_cur = os.path.join(MODEL_OUT_DIR_CUR, "model.onnx")
    modelo_actual_existe = descargar_modelo_actual_minio(onnx_path_cur)

    if modelo_actual_existe:
        cur_metrics = evaluate_onnx_model(onnx_path_cur, test_ds)
        print("🔄 Métricas modelo en producción:", cur_metrics)
    else:
        print("⚠️ No hay modelo actual. Se asumirá accuracy = 0")
        cur_metrics = {'eval_accuracy': 0.0}

    # Comparar y decidir deploy
    if new_metrics["eval_accuracy"] > cur_metrics["eval_accuracy"]:
        print("🎉 Nuevo modelo supera al actual. Subiendo a MinIO…")
        subir_nuevo_modelo_a_minio(onnx_path_new)
    else:
        print("⚠️ Nuevo modelo NO supera al actual. Manteniendo despliegue.")


if __name__ == "__main__":
    main()
