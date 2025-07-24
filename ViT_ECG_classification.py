# !pip install transformers datasets accelerate evaluate albumentations

# !pip install evaluate

import os
import random
import shutil

random.seed(1337)

import ssl

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split

ssl._create_default_https_context = ssl._create_stdlib_context

import torch
import albumentations as A
from datasets import load_dataset
from evaluate import load
from transformers import (
    Trainer,
    TrainingArguments,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
)

try:
  shutil.rmtree("./data/")
except:
  pass
try:
  shutil.rmtree("./vit-base-res/")
except:
  pass

# !pip install --upgrade datasets fsspec


from datasets import load_dataset
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shutil


# Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Ruta del dataset
DATASET_PATH = "/content/drive/MyDrive/Reto_3_IA"

# Cargar dataset sin guardarlo todo en memoria
dataset = load_dataset("imagefolder", data_dir=DATASET_PATH)#, streaming=True)

print(dataset["train"].features["label"].names)


def show_examples(ds, seed: int = 1234, examples_per_class: int = 3, size=(350, 350)):

    w, h = size
    labels = ds["train"].features["label"].names
    grid = Image.new("RGB", size=(examples_per_class * w, len(labels) * h))
    draw = ImageDraw.Draw(grid)

    for label_id, label in enumerate(labels):

        # Filter the dataset by a single label, shuffle it, and grab a few samples
        ds_slice = (
            ds["train"]
            .filter(lambda ex: ex["label"] == label_id)
            .shuffle(seed)
            .select(range(examples_per_class))
        )

        # Plot this label's examples along a row
        for i, example in enumerate(ds_slice):
            image = example["image"]
            idx = examples_per_class * label_id + i
            box = (idx % examples_per_class * w, idx // examples_per_class * h)
            grid.paste(image.resize(size), box=box)
            draw.text(box, label, (255, 255, 255))

    return grid


#show_examples(ds, seed=random.randint(0, 1337), examples_per_class=3)

model_name_or_path = "google/vit-base-patch16-224-in21k"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)

train_transforms = A.Compose([
    A.RandomRotate90(),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    #A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

val_transforms = A.Compose([
    #A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])


def transform(example_batch):
    inputs = feature_extractor([x for x in example_batch["pixel_values"]], return_tensors="pt")
    inputs["label"] = example_batch["label"]
    return inputs

def preprocess_train(examples):
    examples["pixel_values"] = [
        train_transforms(image=np.array(image))["image"] for image in examples["image"]
    ]

    return transform(examples)

def preprocess_val(examples):
    examples["pixel_values"] = [
        val_transforms(image=np.array(image))["image"] for image in examples["image"]
    ]

    return transform(examples)

train_ds = dataset["train"].with_transform(preprocess_train)
test_ds = dataset["test"].with_transform(preprocess_val)

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["label"] for x in batch]),
    }

metric = load("f1")

def compute_metrics(p):
    return metric.compute(
        predictions=np.argmax(p.predictions, axis=1), references=p.label_ids, average="weighted"
    )

labels = dataset["train"].features["label"].names

model = AutoModelForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
    ignore_mismatched_sizes=True
)

training_args = TrainingArguments(
    output_dir="./vit-base-res",
    per_device_train_batch_size=32,

    # Useful parameters to reduce GPU usage
    # gradient_accumulation_steps=4,
    # gradient_checkpointing=True,

    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=10,
    # fp16=True,
    # tf32=True,
    # save_steps=100,
    # eval_steps=100,
    # logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="tensorboard",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=feature_extractor,
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

# Importar la librería
from transformers import ViTForImageClassification

# Ruta donde se guardará el modelo en Google Drive
model_path = "/content/drive/MyDrive/Reto_3_IA/vit_model_2"

# Guardar el modelo en formato Hugging Face
model.save_pretrained(model_path)

# Si tienes un feature extractor o processor, guárdalo también
feature_extractor.save_pretrained(model_path)


metrics = trainer.evaluate(test_ds)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

import torch
import torch.nn.functional as F
from sklearn import metrics

y_preds = []
y_trues = []
print(test_ds)
for idx, data in enumerate(test_ds):
  x = torch.unsqueeze(data["pixel_values"], dim=0).cuda()
  with torch.no_grad():
    logits = model(x).logits
  probability = torch.nn.functional.softmax(logits, dim=-1).cpu()
  probs = probability[0].detach().numpy()
  confidences = {label: float(probs[i]) for i, label in enumerate(labels)}
  y_pred = max(confidences, key=confidences.get)
  y_preds.append(y_pred)
  y_trues.append(data["label"])

y_trues = [str(y) for y in y_trues]
y_preds = [str(y) for y in y_preds]
class_labels = dataset["train"].features["label"].names
label_mapping = {name: idx for idx, name in enumerate(class_labels)}
y_preds = [label_mapping[label] for label in y_preds]
y_trues = [int(y) for y in y_trues]
print(class_labels)
print(label_mapping)
print(y_trues[:10])
print(y_preds[:10])

confusion_matrix = metrics.confusion_matrix(y_trues, y_preds, labels=list(range(len(class_labels))))
print(confusion_matrix)

import seaborn as sns
sns.heatmap(confusion_matrix, annot=True, fmt=".0f", linewidth=.1, cmap="crest")

from sklearn.metrics import classification_report
print(classification_report(y_trues, y_preds, target_names=labels))

image = Image.open('/content/drive/MyDrive/Reto_3_IA/ecg_test.png').convert("RGB")

# Preprocesar la imagen
inputs = feature_extractor(images=image, return_tensors="pt")

# Mover a GPU si está disponible
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Hacer la predicción
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Aplicar softmax para obtener probabilidades
probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()

# Obtener las clases del dataset
class_labels = dataset["train"].features["label"].names

# Obtener la clase con mayor probabilidad
predicted_class_idx = np.argmax(probs)
predicted_label = class_labels[predicted_class_idx]

# Imprimir resultado
print(f"Predicción: {predicted_label} (Confianza: {probs[predicted_class_idx]:.4f})")


# !pip install onnx

import torch
import onnx
from transformers import ViTForImageClassification
from torch import nn


# Ruta del modelo en Google Drive
pth_model_path = "/content/drive/MyDrive/Reto_3_IA/vit_model_2"  # Cambia según tu ruta
onnx_model_path = "/content/drive/MyDrive/vit_model/vit_model.onnx"

# Chequear que la carpeta de salida exista (si no, se crea)
output_dir = os.path.dirname(onnx_model_path)
os.makedirs(output_dir, exist_ok=True)

# Cargar modelo ya con el fine-tuning
model = ViTForImageClassification.from_pretrained(pth_model_path)

# model.classifier = nn.Linear(768, 4)  # Cambia el 4 por el número correcto de clases

# Cargar los pesos entrenados
# model.load_state_dict(torch.load(pth_model_path, map_location="cpu"))

# Poner en modo evaluación
model.eval()

# Crear una entrada de prueba (imagen RGB 224x224)
dummy_input = torch.randn(1, 3, 224, 224)

# Exportar a ONNX
torch.onnx.export(
    model, dummy_input, onnx_model_path,
    export_params=True, opset_version=14,
    do_constant_folding=True,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

print(f"✅ Modelo exportado a ONNX en: {onnx_model_path}")


# !pip install onnxruntime opencv-python

import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

# Ruta al modelo ONNX en Google Drive
ONNX_MODEL_PATH = "/content/drive/MyDrive/vit_model/vit_model.onnx"

# Ruta a la imagen en Google Drive
IMAGE_PATH = "/content/drive/MyDrive/Reto_3_IA/ecg_test.png"

# Cargar el modelo ONNX
ort_session = ort.InferenceSession(ONNX_MODEL_PATH)

# Definir transformaciones de la imagen (ajustar según el modelo)
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Ajusta según el tamaño esperado por ViT
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalización estándar
])

# Cargar y preprocesar la imagen
image = Image.open(IMAGE_PATH).convert("RGB")  # Convertir a RGB
image = transform(image)
image = image.unsqueeze(0).numpy()  # Agregar dimensión batch y convertir a NumPy

# Realizar la inferencia con ONNX Runtime
inputs = {ort_session.get_inputs()[0].name: image}
outputs = ort_session.run(None, inputs)

# Obtener la predicción (clase con mayor probabilidad)
predicted_class = np.argmax(outputs[0])

print(f"🔹 Predicción del modelo ONNX: Clase {predicted_class}")


