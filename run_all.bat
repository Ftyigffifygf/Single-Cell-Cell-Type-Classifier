@echo off
echo === Single-Cell Cell-Type Classifier Pipeline ===
echo Starting complete pipeline execution...

REM Create necessary directories
echo Creating directories...
if not exist data mkdir data
if not exist models mkdir models
if not exist results mkdir results

REM Step 1: Create demo data
echo.
echo Step 1: Creating demo PBMC data...
python create_demo_data.py

REM Step 2: Preprocess data
echo.
echo Step 2: Preprocessing data...
python preprocessing.py

REM Step 3: Generate embeddings
echo.
echo Step 3: Generating Geneformer embeddings...
python embed_geneformer_enhanced.py

REM Step 4: Train classifier
echo.
echo Step 4: Training cell type classifier...
python train_classifier.py

REM Step 5: Evaluate model
echo.
echo Step 5: Evaluating model performance...
python evaluate.py

echo.
echo === Pipeline Complete ===
echo Results saved in:
echo   - Models: models/
echo   - Evaluation: results/
echo   - Data: data/
echo.
echo To start the API server, run:
echo   python serve_api.py

pause
