from fastapi import FastAPI, HTTPException, File, UploadFile
from minio import Minio
import os
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image
import numpy as np
import onnxruntime as ort

load_dotenv()

app = FastAPI()

rootEndpoint = "ecg/api/v1/model"


# Configuración de MinIO desde .env
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET = os.getenv("MINIO_BUCKET")
MINIO_MODEL_OBJECT = os.getenv("MINIO_MODEL_OBJECT")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH")



# Guardar la fecha de modificación localmente, asociada al modelo
MODEL_TIMESTAMP_PATH = LOCAL_MODEL_PATH + ".timestamp"

def descargar_modelo():
    print(f"init descargar")
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=True
    )
    # Obtener metadata del objeto en MinIO
    print(f"antes stat")
    stat = client.stat_object(MINIO_BUCKET, MINIO_MODEL_OBJECT)
    minio_last_modified = stat.last_modified.timestamp()
    print(f"Última modificación en MinIO: {minio_last_modified}")

    # Leer timestamp local
    local_last_modified = None
    if os.path.exists(MODEL_TIMESTAMP_PATH):
        with open(MODEL_TIMESTAMP_PATH, "r") as f:
            try:
                local_last_modified = float(f.read().strip())
            except Exception:
                local_last_modified = None

    # Descargar si no existe o si el modelo cambió
    if (not os.path.exists(LOCAL_MODEL_PATH)) or (local_last_modified != minio_last_modified):
        print(f"Descargando modelo desde MinIO: {MINIO_MODEL_OBJECT} a {LOCAL_MODEL_PATH}")
        client.fget_object(MINIO_BUCKET, MINIO_MODEL_OBJECT, LOCAL_MODEL_PATH)
        print(client)
        print(client.fget_object(MINIO_BUCKET, MINIO_MODEL_OBJECT, LOCAL_MODEL_PATH))
        with open(MODEL_TIMESTAMP_PATH, "w") as f:
            f.write(str(minio_last_modified))



def predecir_imagen_jpg_file(file: UploadFile):
    try:
        image_bytes = file.file.read()
        image = Image.open(BytesIO(image_bytes))
        if image.format not in ["JPEG", "JPG"]:
            raise HTTPException(status_code=400, detail="Formato de imagen no soportado. Solo JPG/JPEG.")
        image = image.convert("RGB")
        image = image.resize((224, 224))  # Redimensionar
        image = np.array(image).astype(np.float32) / 255.0  # Normalizar a [0,1]
        image = (image - 0.5) / 0.5  # Normalizar a [-1, 1]
        image = np.transpose(image, (2, 0, 1))  # Canal primero
        image = np.expand_dims(image, axis=0) 


        descargar_modelo()
        onnx_path = LOCAL_MODEL_PATH
        ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        inputs = {ort_session.get_inputs()[0].name: image}
        outputs = ort_session.run(None, inputs)
        print(f"outputs: {outputs}")
        probs = np.exp(outputs[0]) / np.exp(outputs[0]).sum()
        probs = probs[0].tolist()
        # Devolver por cada clase la probabilidad como lista de dicts
        print(f"Probabilidades len: {len(probs)}")
        CLASS_NAMES = [
            "Normal",
            "Myocardial Infarction",
            "History of MI",
            "Abnormal Heartbeat",
            "Left Ventricular Hypertrophy",
            "Left Anterior Fascicular Block",
        ]
        
        clases = []
        for i, prob in enumerate(probs):
            clases.append({"name": CLASS_NAMES[i], "probability": prob})
        
        return clases
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error procesando la imagen o el modelo: " + str(e))


predictEndpoint = rootEndpoint + "/predict"
@app.post("/ecg/api/v1/model/predict")
def predict_endpoint(file: UploadFile = File(...)):
    """
    Recibe una imagen JPG como archivo y retorna la predicción ONNX.
    """
    clases = predecir_imagen_jpg_file(file)
    return {
        "predictions": clases,
        "heart_rate": None
    }




