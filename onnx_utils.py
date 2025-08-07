# onnx_utils.py
import os
import torch
import onnxruntime as ort
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

def convert_to_onnx(model, output_dir: str, model_filename: str = "model.onnx"):
    """
    Convierte un modelo de PyTorch (Hugging Face) a formato ONNX.

    Args:
        model: El modelo de PyTorch entrenado.
        output_dir: El directorio donde se guardará el archivo ONNX.
        model_filename: El nombre del archivo ONNX.
    """
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, model_filename)
    
    # Poner el modelo en modo de evaluación
    model.eval()
    device = next(model.parameters()).device
    
    # Crear una entrada de prueba (dummy input) con el formato correcto
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True).to(device)

    print(f"⚙️ Exportando modelo a ONNX en: {onnx_path}")
    
    # Exportar el modelo
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "logits": {0: "batch_size"}
        }
    )
    
    print("✅ Modelo exportado a ONNX con éxito.")
    return onnx_path

def evaluate_onnx_model(onnx_path: str, test_dataset):
    """
    Evalúa un modelo ONNX en un conjunto de datos de prueba.

    Args:
        onnx_path: La ruta al archivo .onnx.
        test_dataset: El dataset de prueba (debe tener 'pixel_values' y 'labels').

    Returns:
        Un diccionario con las métricas de evaluación.
    """
    print(f"Evaluando modelo ONNX desde: {onnx_path}")
    
    # Cargar el modelo ONNX
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    all_preds = []
    all_labels = []

    # Iterar sobre el dataset de prueba
    for item in tqdm(test_dataset, desc="Evaluando ONNX"):
        pixel_values = np.expand_dims(item['pixel_values'], axis=0)
        labels = item['labels']
        
        # Realizar la inferencia
        ort_inputs = {input_name: pixel_values}
        ort_outs = ort_session.run(None, ort_inputs)
        
        # Obtener la predicción
        pred = np.argmax(ort_outs[0], axis=1)[0]
        all_preds.append(pred)
        all_labels.append(labels)

    # Calcular métricas
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    
    metrics = {
        "eval_accuracy": accuracy,
        "eval_f1": f1
    }
    
    return metrics