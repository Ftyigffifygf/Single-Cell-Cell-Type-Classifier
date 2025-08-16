from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import toml
import torch
from typing import List, Dict, Any
import uvicorn
from embed_geneformer import GeneformerEmbedder

# Load configuration
def load_config():
    with open('config.toml', 'r') as f:
        return toml.load(f)

config = load_config()

# Initialize FastAPI app
app = FastAPI(title="Single-Cell Cell-Type Classifier API", version="1.0.0")

# Global variables for model components
model_bundle = None
embedder = None

class CellData(BaseModel):
    gene_expression: List[float]
    gene_names: List[str]

class PredictionResponse(BaseModel):
    cell_type: str
    confidence: float
    probabilities: Dict[str, float]

@app.on_event("startup")
async def load_models():
    """Load model components on startup"""
    global model_bundle, embedder
    
    try:
        # Load trained classifier bundle
        print(f"Loading model bundle from {config['training']['model_path']}")
        model_bundle = joblib.load(config['training']['model_path'])
        
        # Initialize Geneformer embedder
        print(f"Loading Geneformer model from {config['model']['geneformer_ckpt']}")
        embedder = GeneformerEmbedder(config['model']['geneformer_ckpt'])
        
        print("Models loaded successfully!")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        raise e

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Single-Cell Cell-Type Classifier API", "status": "healthy"}

@app.get("/info")
async def get_model_info():
    """Get model information"""
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "feature_dimension": model_bundle['feature_dim'],
        "classes": model_bundle['classes'],
        "num_classes": len(model_bundle['classes']),
        "model_type": type(model_bundle['classifier']).__name__
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_cell_type(cell_data: CellData):
    """Predict cell type from gene expression data"""
    
    if model_bundle is None or embedder is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Validate input
        if len(cell_data.gene_expression) != len(cell_data.gene_names):
            raise HTTPException(
                status_code=400, 
                detail="Gene expression and gene names must have the same length"
            )
        
        # Convert to numpy array
        expression_array = np.array(cell_data.gene_expression)
        
        # Generate embedding using Geneformer
        if embedder.model is None:
            # Use random embedding for demo
            embedding = np.random.randn(768)
        else:
            # Convert to tokens
            tokens = embedder.genes_to_tokens(expression_array, cell_data.gene_names)
            
            if not tokens:
                embedding = np.zeros(768)
            else:
                # Tokenize and encode
                inputs = embedder.tokenizer(
                    ' '.join(tokens),
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                
                inputs = {k: v.to(embedder.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = embedder.model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        
        # Scale embedding
        embedding_scaled = model_bundle['scaler'].transform(embedding.reshape(1, -1))
        
        # Make prediction
        prediction = model_bundle['classifier'].predict(embedding_scaled)[0]
        probabilities = model_bundle['classifier'].predict_proba(embedding_scaled)[0]
        
        # Convert to cell type name
        cell_type = model_bundle['label_encoder'].inverse_transform([prediction])[0]
        confidence = float(np.max(probabilities))
        
        # Create probability dictionary
        prob_dict = {}
        for i, class_name in enumerate(model_bundle['classes']):
            prob_dict[class_name] = float(probabilities[i])
        
        return PredictionResponse(
            cell_type=cell_type,
            confidence=confidence,
            probabilities=prob_dict
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(cells_data: List[CellData]):
    """Predict cell types for multiple cells"""
    
    if model_bundle is None or embedder is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        predictions = []
        
        for cell_data in cells_data:
            # This is a simplified batch processing - in production, 
            # you'd want to optimize this for true batch processing
            result = await predict_cell_type(cell_data)
            predictions.append(result)
        
        return {"predictions": predictions}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host=config['api']['host'], 
        port=config['api']['port']
    )
