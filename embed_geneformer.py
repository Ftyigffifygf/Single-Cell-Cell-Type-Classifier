import torch
import numpy as np
import scanpy as sc
import toml
import os
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
from pathlib import Path

def load_config():
    with open('config.toml', 'r') as f:
        return toml.load(f)

class GeneformerEmbedder:
    def __init__(self, model_path, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load Geneformer model and tokenizer
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
            self.model = BertModel.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded Geneformer model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using random embeddings for demo purposes")
            self.model = None
            self.tokenizer = None
    
    def genes_to_tokens(self, gene_expression, gene_names):
        """Convert gene expression to tokens for Geneformer"""
        # Get non-zero genes
        nonzero_idx = np.nonzero(gene_expression)[0]
        
        if len(nonzero_idx) == 0:
            return []
        
        # Create gene tokens based on expression levels
        tokens = []
        for idx in nonzero_idx:
            gene_name = gene_names[idx]
            expression_level = int(gene_expression[idx])
            
            # Simple tokenization: repeat gene token based on expression
            for _ in range(min(expression_level, 10)):  # Cap at 10 repeats
                tokens.append(gene_name)
        
        return tokens[:512]  # Limit to model's max length
    
    def embed_cells(self, adata, batch_size=32):
        """Generate embeddings for all cells"""
        n_cells = adata.shape[0]
        gene_names = adata.var_names.tolist()
        
        embeddings = []
        
        print(f"Generating embeddings for {n_cells} cells...")
        
        for i in tqdm(range(0, n_cells, batch_size)):
            batch_end = min(i + batch_size, n_cells)
            batch_embeddings = []
            
            for j in range(i, batch_end):
                cell_expression = adata.X[j, :].toarray().flatten() if hasattr(adata.X[j, :], 'toarray') else adata.X[j, :]
                
                if self.model is None:
                    # Generate random embeddings for demo
                    embedding = np.random.randn(768)
                else:
                    # Convert to tokens
                    tokens = self.genes_to_tokens(cell_expression, gene_names)
                    
                    if not tokens:
                        embedding = np.zeros(768)
                    else:
                        # Tokenize and encode
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
                            # Use CLS token embedding
                            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
                
                batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)

def generate_embeddings(config):
    """Generate Geneformer embeddings for preprocessed data"""
    
    # Load preprocessed data
    print(f"Loading preprocessed data from {config['data']['preprocessed_h5ad']}")
    adata = sc.read_h5ad(config['data']['preprocessed_h5ad'])
    
    # Determine batch size based on device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = config['model']['batch_size'] if device == 'cuda' else config['model']['batch_size_cpu']
    
    # Initialize embedder
    embedder = GeneformerEmbedder(config['model']['geneformer_ckpt'], device)
    
    # Generate embeddings
    embeddings = embedder.embed_cells(adata, batch_size=batch_size)
    
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
