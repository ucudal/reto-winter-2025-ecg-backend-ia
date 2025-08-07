from transformers import Trainer
from hf_utils import collate_fn

# en desuso, evaluaba el modelo de pytorch y usamos un onnx
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
