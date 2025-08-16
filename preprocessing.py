import scanpy as sc
import pandas as pd
import numpy as np
import toml
import os
from pathlib import Path

def load_config():
    with open('config.toml', 'r') as f:
        return toml.load(f)

def preprocess_data(config):
    input_path = config['data']['input_h5ad']
    print(f'Loading data from {input_path}')
    adata = sc.read_h5ad(input_path)
    
    print(f'Original data shape: {adata.shape}')
    print('Label distribution:')
    print(adata.obs[config['data']['label_column']].value_counts())
    
    # Make variable names unique
    adata.var_names_make_unique()
    
    # Filter cells and genes
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Calculate QC metrics
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    
    # Filter cells with too many mitochondrial genes (if column exists)
    if 'pct_counts_mt' in adata.obs.columns:
        adata = adata[adata.obs.pct_counts_mt < 20, :]
    
    # Keep raw counts for Geneformer
    adata.raw = adata
    
    # Normalize and log transform for gene selection
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Find highly variable genes
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    # Select top genes per cell based on expression
    top_genes = config['model']['top_genes']
    
    # Get raw counts back
    adata = adata.raw.to_adata()
    
    # For each cell, select top expressed genes
    def select_top_genes_per_cell(X, n_genes):
        result = np.zeros_like(X)
        for i in range(X.shape[0]):
            cell_data = X[i, :].toarray().flatten() if hasattr(X[i, :], 'toarray') else X[i, :]
            top_indices = np.argsort(cell_data)[-n_genes:]
            result[i, top_indices] = cell_data[top_indices]
        return result
    
    print(f'Selecting top {top_genes} genes per cell...')
    adata.X = select_top_genes_per_cell(adata.X, top_genes)
    
    print(f'Processed data shape: {adata.shape}')
    
    # Create output directory
    output_path = config['data']['preprocessed_h5ad']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save preprocessed data
    adata.write_h5ad(output_path)
    print(f'Saved preprocessed data to {output_path}')
    
    return adata

if __name__ == '__main__':
    config = load_config()
    preprocess_data(config)
