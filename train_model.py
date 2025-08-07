import albumentations as A
from transformers import (
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments
)
from hf_utils import make_ds, preprocess, collate_fn, compute_metrics

def train_ecg_model(
    train_docs,  # lista de dicts con "image": np.array, "label": int
    val_docs,
    model_out_dir: str,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 2e-4,
):
    """
    Entrena un ViT con train_docs/val_docs, guarda en model_out_dir
    y devuelve el objeto Trainer.model.
    """
    ds_train = make_ds(train_docs)
    ds_val   = make_ds(val_docs)

    model_name = "google/vit-base-patch16-224-in21k"
    
    # feature extractor + transformaciones
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    train_tf = A.Compose([
        A.RandomRotate90(),
        A.HorizontalFlip(0.5),
        A.RandomBrightnessContrast(0.2),
    ])
    val_tf = A.Compose([])

    train_ds = ds_train.with_transform(lambda b: preprocess(b, train_tf, extractor))
    val_ds   = ds_val.with_transform(lambda b: preprocess(b, val_tf, extractor))

    # modelo y trainer
    unique = sorted({d["label"] for d in train_docs + val_docs})
    id2lab = {i:str(i) for i in unique}
    lab2id = {str(i):i for i in unique}

    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=len(unique),
        id2label=id2lab,
        label2id=lab2id,
        ignore_mismatched_sizes=True
    )

    args = TrainingArguments(
        output_dir=model_out_dir,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=extractor,
    )

    # train & save
    trainer.train()
    trainer.save_model(model_out_dir)
    extractor.save_pretrained(model_out_dir)

    return trainer.model
