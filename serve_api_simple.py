from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import toml
import os
from typing import List, Dict
import uvicorn

# Load configuration
config = toml.load('config.toml')

# Initialize FastAPI app
app = FastAPI(title='Single-Cell Cell-Type Classifier API', version='1.0.0')

# Load model on startup
model_bundle = joblib.load('models/ct_head.joblib')

class CellData(BaseModel):
    gene_expression: List[float]
    gene_names: List[str]

class PredictionResponse(BaseModel):
    cell_type: str
    confidence: float
    probabilities: Dict[str, float]

@app.get('/')
async def root():
    return {'message': 'Single-Cell Cell-Type Classifier API', 'status': 'healthy'}

@app.get('/info')
async def get_model_info():
    return {
        'feature_dimension': model_bundle['feature_dim'],
        'classes': model_bundle['classes'],
        'num_classes': len(model_bundle['classes']),
        'model_type': type(model_bundle['classifier']).__name__
    }

@app.post('/predict', response_model=PredictionResponse)
async def predict_cell_type(cell_data: CellData):
    try:
        # Validate input
        if len(cell_data.gene_expression) != len(cell_data.gene_names):
            raise HTTPException(status_code=400, detail='Gene expression and gene names must have same length')
        
        # For demo: create random embedding (in production, use Geneformer)
        embedding = np.random.randn(1024)  # Match NVIDIA embedding dimension
        
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
        raise HTTPException(status_code=500, detail=f'Prediction error: {str(e)}')

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
