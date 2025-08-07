from datasets import Dataset, Image as HfImage
from PIL import Image
import numpy as np
import torch
from evaluate import load as load_metric

metric = load_metric("f1")
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return metric.compute(predictions=preds, references=p.label_ids, average="weighted")

def make_ds(docs):
    images = []
    labels = []
    for d in docs:
        img = Image.open(d["image"])
        images.append(img)
        labels.append(d["label"])
    
    dataset = Dataset.from_dict({
        "image": images,
        "label": labels
    })
    
    return dataset.cast_column("image", HfImage(decode=True))

def preprocess(batch, tfm, extractor):
        imgs = [tfm(image=np.array(img))["image"] for img in batch["image"]] # agrega transformaciones a cada imagen
        enc = extractor(imgs, return_tensors="pt") # se redimensionan y normalizan las imágenes / pt = pytorch tensors
        enc["labels"] = torch.tensor(batch["label"]) # agrega etiquetas como tensores
        return enc

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }

def build_test_dataset(test_docs, extractor, val_tf):
    """
    Construye un HF Dataset para test:
      - cast image
      - apply extractor + val_tf
    """
    ds = make_ds(test_docs)
    return ds.with_transform(lambda b: preprocess(b, val_tf, extractor))