from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import asyncio
import traceback

app = FastAPI(
    title="ECG IA Training API",
    description="API for training Vision Transformer models on ECG data",
    version="1.0.0"
)

# Pydantic model for the training request
class TrainingRequest(BaseModel):
    epochs: int = 10
    learning_rate: float = 2e-4
    batch_size: int = None  # Will auto-detect optimal batch size if None
    
class TrainingResponse(BaseModel):
    success: bool
    message: str
    training_log: Dict[str, Any] = None
    error: str = None

@app.get("/")
async def root():
    return {"message": "ECG IA Training API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ECG IA Training API"}

@app.post("/model/full-retrain", response_model=TrainingResponse)
async def start_full_retrain(request: TrainingRequest = None):
    """
    Start a full retraining of the Vision Transformer model.
    
    This endpoint will:
    1. Load data from MongoDB
    2. Train a new ViT model using the pipeline
    3. Evaluate and compare with current model
    4. Deploy to S3 if performance improves
    
    Returns:
        TrainingResponse indicating if training was started successfully (not waiting for completion)
    """
    try:
        # Import the main pipeline function
        from pipeline import main
        
        # Start the training pipeline in a thread pool without waiting for completion
        # For production, consider using Celery or similar for background tasks
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, main)
        
        return TrainingResponse(
            success=True,
            message="Training pipeline started successfully"
        )
        
    except ImportError as e:
        return TrainingResponse(
            success=False,
            message="Pipeline module not found",
            error=str(e)
        )
    except Exception as e:
        error_traceback = traceback.format_exc()
        return TrainingResponse(
            success=False,
            message="Failed to start training pipeline",
            error=f"{str(e)}\n\n{error_traceback}"
        )

@app.get("/models")
async def list_models():
    """List all available trained models"""
    import os
    models_dir = "./models"
    if not os.path.exists(models_dir):
        return {"models": []}
    
    models = []
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path) and item.startswith("vit_model_"):
            models.append(item)
        elif item.endswith(".onnx"):
            models.append(item)
    
    return {"models": models}

@app.get("/training-logs")
async def list_training_logs():
    """List all available training logs"""
    import os
    logs_dir = "./traininglogs"
    if not os.path.exists(logs_dir):
        return {"logs": []}
    
    logs = []
    for item in os.listdir(logs_dir):
        if item.endswith(".json"):
            logs.append(item)
    
    return {"logs": logs}

@app.get("/training-logs/{log_filename}")
async def get_training_log(log_filename: str):
    """Get a specific training log"""
    import os
    import json
    
    log_path = os.path.join("./traininglogs", log_filename)
    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="Training log not found")
    
    try:
        with open(log_path, 'r') as f:
            log_data = json.load(f)
        return log_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading log file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
