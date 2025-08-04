from transformers import Trainer
from hf_utils import make_ds, preprocess, collate_fn

def build_test_dataset(test_docs, extractor, val_tf):
    """
    Construye un HF Dataset para test:
      - cast image
      - apply extractor + val_tf
    """
    ds = make_ds(test_docs)
    return ds.with_transform(lambda b: preprocess(b, val_tf, extractor))

def evaluate_model(model, test_dataset, extractor, val_tf, batch_size: int = 32):
    """
    Evalúa `model` sobre `test_dataset`.
    Devuelve el dict de métricas (p.ej. 'eval_accuracy').
    """
    trainer = Trainer(
        model=model,
        tokenizer=extractor,
        data_collator=collate_fn,
    )
    return trainer.evaluate(eval_dataset=test_dataset, batch_size=batch_size)
