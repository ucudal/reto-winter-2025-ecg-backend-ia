# Local ViT Training Script
# Make sure you have train/ and test/ folders in your working directory

# !pip install transformers datasets accelerate evaluate albumentations torch torchvision scikit-learn seaborn matplotlib pillow numpy opencv-python tensorboard

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

# GPU Setup and Optimization
print("Setting up GPU and performance optimizations...")
if torch.cuda.is_available():
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    device = torch.device("cuda")
    
    # Enable optimizations for GPU
    torch.backends.cudnn.benchmark = True  # Optimize cudnn for consistent input sizes
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster training on Ampere GPUs
    torch.backends.cudnn.allow_tf32 = True
else:
    print("❌ GPU not available, using CPU (training will be much slower)")
    device = torch.device("cpu")

# Set number of threads for CPU operations
torch.set_num_threads(os.cpu_count())

def get_optimal_batch_size():
    """Automatically determine optimal batch size based on GPU memory"""
    if not torch.cuda.is_available():
        return 8  # Conservative batch size for CPU
    
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    if gpu_memory_gb >= 24:  # RTX 4090, A100, etc.
        return 128
    elif gpu_memory_gb >= 16:  # RTX 4080, RTX 3090, etc.
        return 96
    elif gpu_memory_gb >= 12:  # RTX 4070 Ti, RTX 3080 Ti, etc.
        return 64
    elif gpu_memory_gb >= 8:   # RTX 4060 Ti, RTX 3070, etc.
        return 32
    else:  # RTX 3060, GTX 1660, etc.
        return 16

# Get optimal batch size
optimal_batch_size = get_optimal_batch_size()
print(f"Using batch size: {optimal_batch_size}")

# Clean up previous runs
try:
    shutil.rmtree("./data/")
except:
    pass
try:
    shutil.rmtree("./vit-base-res/")
except:
    pass

# Local dataset path - assumes train/ and test/ folders exist in current directory
DATASET_PATH = "."

# Load dataset from local folders
dataset = load_dataset("imagefolder", data_dir=DATASET_PATH)

print("Classes found:", dataset["train"].features["label"].names)

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

# Uncomment to show examples
# show_examples(dataset, seed=random.randint(0, 1337), examples_per_class=3)

model_name_or_path = "google/vit-base-patch16-224-in21k"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)

train_transforms = A.Compose([
    # Geometric transforms
    A.RandomRotate90(p=0.3),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.3),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=0.3),
    
    # Color/Intensity transforms
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.CLAHE(clip_limit=2.0, p=0.2),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.2),
    
    # Noise and blur
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 30.0)),
        A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3)),
    ], p=0.2),
    
    A.OneOf([
        A.GaussianBlur(blur_limit=(1, 3)),
        A.MotionBlur(blur_limit=3),
    ], p=0.1),
    
    # Distortions
    A.OneOf([
        A.GridDistortion(num_steps=5, distort_limit=0.1),
        A.OpticalDistortion(distort_limit=0.1, shift_limit=0.05),
        A.ElasticTransform(alpha=1, sigma=50),
    ], p=0.2),
    
    # Cutout
    A.CoarseDropout(max_holes=3, max_height=16, max_width=16, p=0.1),
    
    # Note: Normalization is handled by the feature_extractor
])

val_transforms = A.Compose([
    # Validation should typically use minimal or no augmentation
    # to get consistent evaluation results
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

# Move model to GPU if available
if torch.cuda.is_available():
    model = model.to(device)
    print(f"✅ Model moved to GPU: {device}")

training_args = TrainingArguments(
    output_dir="./vit-base-res",
    per_device_train_batch_size=optimal_batch_size,
    per_device_eval_batch_size=optimal_batch_size,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="none",
    load_best_model_at_end=True,
    
    # GPU Optimization settings
    fp16=True,                       # Use mixed precision training (faster + less memory)
    tf32=True,                     # Use TensorFloat-32 on Ampere GPUs (NVIDIA only)
    dataloader_num_workers=min(8, os.cpu_count()),  # Optimal number of workers
    gradient_checkpointing=True,     # Trade compute for memory (allows larger batch sizes)
    gradient_accumulation_steps=1,   # Accumulate gradients
    
    # Logging and saving optimizations
    logging_steps=50,                # Log less frequently
    save_steps=500,                  # Save less frequently
    eval_steps=500,                  # Evaluate less frequently during training
    
    # Performance optimizations
    dataloader_pin_memory=True,      # Pin memory for faster GPU transfer
    group_by_length=False,           # Don't group by length for vision tasks
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

print("Starting training...")
train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

# Save model locally
from transformers import ViTForImageClassification

# Local path to save the model
model_path = "./saved_vit_model"
os.makedirs(model_path, exist_ok=True)

# Save the model in Hugging Face format
model.save_pretrained(model_path)
feature_extractor.save_pretrained(model_path)

print(f"Model saved to: {model_path}")

# Evaluate the model
print("Evaluating model...")
metrics = trainer.evaluate(test_ds)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

# Detailed evaluation with confusion matrix
import torch.nn.functional as F
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

y_preds = []
y_trues = []

print("Generating predictions for confusion matrix...")
for idx, data in enumerate(test_ds):
    x = torch.unsqueeze(data["pixel_values"], dim=0)
    if torch.cuda.is_available():
        x = x.cuda()
    
    with torch.no_grad():
        logits = model(x).logits
    
    probability = torch.nn.functional.softmax(logits, dim=-1).cpu()
    probs = probability[0].detach().numpy()
    confidences = {label: float(probs[i]) for i, label in enumerate(labels)}
    y_pred = max(confidences, key=confidences.get)
    y_preds.append(y_pred)
    y_trues.append(data["label"])

# Convert predictions for confusion matrix
y_trues = [str(y) for y in y_trues]
y_preds = [str(y) for y in y_preds]
class_labels = dataset["train"].features["label"].names
label_mapping = {name: idx for idx, name in enumerate(class_labels)}
y_preds = [label_mapping[label] for label in y_preds]
y_trues = [int(y) for y in y_trues]

print("Classes:", class_labels)
print("Label mapping:", label_mapping)

# Generate confusion matrix
confusion_matrix = metrics.confusion_matrix(y_trues, y_preds, labels=list(range(len(class_labels))))
print("Confusion Matrix:")
print(confusion_matrix)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, fmt=".0f", linewidth=.1, cmap="crest", 
            xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

# Classification report
from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(y_trues, y_preds, target_names=labels))

# Test with a single image (replace with your test image path)
def test_single_image(image_path):
    """Test the model with a single image"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    image = Image.open(image_path).convert("RGB")
    
    # Preprocess the image
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
    
    # Get classes from dataset
    class_labels = dataset["train"].features["label"].names
    
    # Get class with highest probability
    predicted_class_idx = np.argmax(probs)
    predicted_label = class_labels[predicted_class_idx]
    
    # Print result
    print(f"Prediction: {predicted_label} (Confidence: {probs[predicted_class_idx]:.4f})")
    
    # Show top 3 predictions
    top_3_indices = np.argsort(probs)[-3:][::-1]
    print("\nTop 3 predictions:")
    for i, idx in enumerate(top_3_indices):
        print(f"{i+1}. {class_labels[idx]}: {probs[idx]:.4f}")

# Example usage (uncomment and provide a test image path)
# test_single_image("path_to_your_test_image.jpg")

# Export to ONNX
def export_to_onnx():
    """Export the trained model to ONNX format"""
    try:
        import onnx
        
        # Create output directory
        onnx_output_dir = "./onnx_model"
        os.makedirs(onnx_output_dir, exist_ok=True)
        onnx_model_path = os.path.join(onnx_output_dir, "vit_model.onnx")
        
        # Load model for export
        model_for_export = ViTForImageClassification.from_pretrained(model_path)
        model_for_export.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Export to ONNX
        torch.onnx.export(
            model_for_export, dummy_input, onnx_model_path,
            export_params=True, opset_version=14,
            do_constant_folding=True,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        
        print(f"✅ Model exported to ONNX: {onnx_model_path}")
        return onnx_model_path
        
    except ImportError:
        print("❌ ONNX not installed. Install with: pip install onnx")
        return None

# Test ONNX model
def test_onnx_model(onnx_path, image_path):
    """Test the ONNX model with an image"""
    try:
        import onnxruntime as ort
        from torchvision import transforms
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return
        
        # Load ONNX model
        ort_session = ort.InferenceSession(onnx_path)
        
        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        image = image.unsqueeze(0).numpy()
        
        # Run inference
        inputs = {ort_session.get_inputs()[0].name: image}
        outputs = ort_session.run(None, inputs)
        
        # Get prediction
        predicted_class = np.argmax(outputs[0])
        confidence = np.max(outputs[0])
        
        print(f"🔹 ONNX Model Prediction: Class {predicted_class} (Confidence: {confidence:.4f})")
        
    except ImportError:
        print("❌ ONNX Runtime not installed. Install with: pip install onnxruntime")

# Uncomment to export and test ONNX model
onnx_path = export_to_onnx()
if onnx_path:
    test_onnx_model(onnx_path, "path_to_your_test_image.jpg")

print("Training completed! Check the following outputs:")
print(f"- Model saved in: {model_path}")
print("- Training logs in: ./vit-base-res")
print("- Confusion matrix saved as: confusion_matrix.png")

print("\n🚀 Performance Tips:")
print("1. Increase batch size if you have more GPU memory")
print("2. Use fp16=True for faster training (already enabled)")
print("3. Set dataloader_num_workers based on your CPU cores")
print("4. Consider using gradient_accumulation_steps for larger effective batch sizes")
print("5. Monitor GPU utilization with: nvidia-smi")

if torch.cuda.is_available():
    print(f"\n📊 GPU Memory Usage:")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")