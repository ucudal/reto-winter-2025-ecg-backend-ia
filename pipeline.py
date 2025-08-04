# BY CHATGPT 

import os
import boto3
import albumentations as A
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

from data_utils   import load_data_from_mongo, get_splits_fixed
from train_model  import train_ecg_model
from eval_model   import build_test_dataset, evaluate_model

def main():
    # ——— Parámetros ———
    MONGO_URI         = "mongodb://root:hK43CrNUq1@mongodb.reto-ucu.net:50005/?authSource=admin"
    DB_NAME           = "admin"
    COLL_NAME         = "ecg_collection"
    TEST_IDS_PATH     = "test_ids.json"
    MODEL_OUT_DIR_NEW = "./models/new"
    MODEL_OUT_DIR_CUR = "./models/current"
    TEST_SIZE, VAL_SIZE, SEED = 0.2, 0.1, 42

    # 1) Carga y split fijo
    docs = load_data_from_mongo(MONGO_URI, DB_NAME, COLL_NAME)
    train_docs, val_docs, test_docs = get_splits_fixed(
        docs, TEST_IDS_PATH, TEST_SIZE, VAL_SIZE, SEED
    )

    # 2) Entrenamiento
    new_model = train_ecg_model(
        train_docs, val_docs,
        model_out_dir=MODEL_OUT_DIR_NEW,
        num_epochs=10,
        batch_size=32,
        learning_rate=2e-4
    )

    # 3) Preparar extractor y val_tf
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_OUT_DIR_NEW)
    val_tf    = A.Compose([])

    # 4) Test dataset
    test_ds = build_test_dataset(test_docs, extractor, val_tf)

    # 5) Evaluar nuevo modelo
    new_metrics = evaluate_model(new_model, test_ds, extractor, val_tf)
    print("✅ Métricas nuevo modelo:", new_metrics)

    # 6) Evaluar modelo actual
    current_model = AutoModelForImageClassification.from_pretrained(MODEL_OUT_DIR_CUR)
    cur_metrics   = evaluate_model(current_model, test_ds, extractor, val_tf)
    print("🔄 Métricas modelo en producción:", cur_metrics)

    # 7) Comparar y deploy a S3 si mejora
    if new_metrics["eval_accuracy"] > cur_metrics["eval_accuracy"]:
        print("🎉 Nuevo modelo supera al actual. Subiendo a S3…")
        s3 = boto3.client("s3")
        for fname in os.listdir(MODEL_OUT_DIR_NEW):
            s3.upload_file(
                os.path.join(MODEL_OUT_DIR_NEW, fname),
                "mi-bucket-modelos",
                f"vit_ecg/{fname}"
            )
    else:
        print("⚠️ Nuevo modelo NO supera al actual. Manteniendo despliegue.")

if __name__ == "__main__":
    main()
