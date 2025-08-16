import torch
import numpy as np
import scanpy as sc
import toml
import os
import requests
import json
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_config():
    with open('config.toml', 'r') as f:
        return toml.load(f)

class EnhancedGeneformerEmbedder:
    def __init__(self, model_path, device=None, use_nvidia_api=True):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_nvidia_api = use_nvidia_api and os.getenv('NVIDIA_API_KEY')
        
        print(f"Using device: {self.device}")
        print(f"NVIDIA API available: {self.use_nvidia_api}")
        
        if self.use_nvidia_api:
            self.nvidia_api_key = os.getenv('NVIDIA_API_KEY')
            self.nvidia_base_url = "https://integrate.api.nvidia.com/v1"
            print("Using NVIDIA API for enhanced embeddings")
        
        # Load Geneformer model and tokenizer as fallback
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
            self.model = BertModel.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded Geneformer model from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load Geneformer model: {e}")
            print("Will use NVIDIA API or random embeddings")
            self.model = None
            self.tokenizer = None
    
    def genes_to_text(self, gene_expression, gene_names):
        """Convert gene expression to text description for NVIDIA API"""
        # Get top expressed genes
        nonzero_idx = np.nonzero(gene_expression)[0]
        
        if len(nonzero_idx) == 0:
            return "No gene expression detected"
        
        # Sort by expression level
        sorted_indices = nonzero_idx[np.argsort(gene_expression[nonzero_idx])[::-1]]
        
        # Create text description
        top_genes = []
        for idx in sorted_indices[:50]:  # Top 50 genes
            gene_name = gene_names[idx]
            expression_level = gene_expression[idx]
            top_genes.append(f"{gene_name}({expression_level:.1f})")
        
        text = f"Single cell gene expression profile: {', '.join(top_genes)}"
        return text[:2000]  # Limit text length
    
    def get_nvidia_embedding(self, text):
        """Get embedding using NVIDIA API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.nvidia_api_key}",
                "Content-Type": "application/json"
            }
            
            # Use NVIDIA's embedding model
            payload = {
                "model": "nvidia/nv-embedqa-e5-v5",
                "input": [text],
                "input_type": "query"
            }
            
            response = requests.post(
                f"{self.nvidia_base_url}/embeddings",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = np.array(result['data'][0]['embedding'])
                return embedding
            else:
                print(f"NVIDIA API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Error calling NVIDIA API: {e}")
            return None
    
    def genes_to_tokens(self, gene_expression, gene_names):
        """Convert gene expression to tokens for Geneformer (fallback)"""
        nonzero_idx = np.nonzero(gene_expression)[0]
        
        if len(nonzero_idx) == 0:
            return []
        
        tokens = []
        for idx in nonzero_idx:
            gene_name = gene_names[idx]
            expression_level = int(gene_expression[idx])
            
            for _ in range(min(expression_level, 10)):
                tokens.append(gene_name)
        
        return tokens[:512]
    
    def embed_single_cell(self, cell_expression, gene_names):
        """Generate embedding for a single cell"""
        
        # Try NVIDIA API first
        if self.use_nvidia_api:
            text = self.genes_to_text(cell_expression, gene_names)
            embedding = self.get_nvidia_embedding(text)
            if embedding is not None:
                return embedding
        
        # Fallback to Geneformer
        if self.model is not None:
            tokens = self.genes_to_tokens(cell_expression, gene_names)
            
            if not tokens:
                return np.zeros(768)
            
            inputs = self.tokenizer(
                ' '.join(tokens),
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
            
            return embedding
        
        # Final fallback: random embedding
        print("Warning: Using random embedding")
        return np.random.randn(1024)  # NVIDIA embeddings are typically 1024-dim
    
    def embed_cells(self, adata, batch_size=32):
        """Generate embeddings for all cells"""
        n_cells = adata.shape[0]
        gene_names = adata.var_names.tolist()
        
        embeddings = []
        
        print(f"Generating embeddings for {n_cells} cells...")
        
        for i in tqdm(range(n_cells)):
            cell_expression = adata.X[i, :].toarray().flatten() if hasattr(adata.X[i, :], 'toarray') else adata.X[i, :]
            embedding = self.embed_single_cell(cell_expression, gene_names)
            embeddings.append(embedding)
            
            # Add small delay for API rate limiting
            if self.use_nvidia_api and i % 10 == 0:
                import time
                time.sleep(0.1)
        
        return np.array(embeddings)

def generate_embeddings(config):
    """Generate enhanced embeddings for preprocessed data"""
    
    # Load preprocessed data
    print(f"Loading preprocessed data from {config['data']['preprocessed_h5ad']}")
    adata = sc.read_h5ad(config['data']['preprocessed_h5ad'])
    
    # Initialize enhanced embedder
    embedder = EnhancedGeneformerEmbedder(
        config['model']['geneformer_ckpt'], 
        use_nvidia_api=True
    )
    
    # Generate embeddings
    embeddings = embedder.embed_cells(adata)
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Create output directory
    os.makedirs(os.path.dirname(config['data']['embeddings_npz']), exist_ok=True)
    
    # Save embeddings and labels
    labels = adata.obs[config['data']['label_column']].values
    
    np.savez_compressed(
        config['data']['embeddings_npz'],
        embeddings=embeddings,
        labels=labels,
        cell_ids=adata.obs_names.values
    )
    
    print(f"Saved embeddings to {config['data']['embeddings_npz']}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Number of cells: {embeddings.shape[0]}")
    print(f"Label distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {label}: {count}")

if __name__ == "__main__":
    config = load_config()
    generate_embeddings(config)
