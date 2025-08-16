import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import os
from pathlib import Path

def create_demo_pbmc_data():
    """Create synthetic PBMC-like data for testing"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define cell types and their characteristics
    cell_types = {
        'T_cells': {'n_cells': 500, 'marker_genes': ['CD3D', 'CD3E', 'CD8A', 'CD4']},
        'B_cells': {'n_cells': 200, 'marker_genes': ['CD19', 'MS4A1', 'CD79A', 'CD79B']},
        'NK_cells': {'n_cells': 150, 'marker_genes': ['KLRD1', 'KLRF1', 'NCR1', 'FCGR3A']},
        'Monocytes': {'n_cells': 300, 'marker_genes': ['CD14', 'FCGR1A', 'CSF1R', 'LYZ']},
        'Dendritic_cells': {'n_cells': 100, 'marker_genes': ['FCER1A', 'CST3', 'CLEC4C']},
        'Platelets': {'n_cells': 50, 'marker_genes': ['PPBP', 'PF4', 'TUBB1']}
    }
    
    # Create gene list (common PBMC genes)
    housekeeping_genes = [
        'ACTB', 'GAPDH', 'RPL13A', 'RPS18', 'HPRT1', 'TBP', 'YWHAZ',
        'B2M', 'UBC', 'PPIA', 'RPL32', 'RPS27A', 'GUSB', 'TFRC'
    ]
    
    # Add more realistic gene names
    additional_genes = [
        'IL2', 'IL4', 'IL6', 'IL10', 'TNF', 'IFNG', 'CCL2', 'CCL3', 'CCL4',
        'CXCL8', 'CXCL10', 'MYC', 'TP53', 'VEGFA', 'EGFR', 'PDGFRA',
        'KIT', 'FLT3', 'CSF2', 'CSF3', 'EPO', 'TGFB1', 'BMP2', 'WNT3A'
    ]
    
    # Collect all marker genes
    all_marker_genes = []
    for cell_type_info in cell_types.values():
        all_marker_genes.extend(cell_type_info['marker_genes'])
    
    # Create comprehensive gene list
    all_genes = list(set(housekeeping_genes + additional_genes + all_marker_genes))
    
    # Add random genes to reach 1000+ genes
    for i in range(len(all_genes), 1200):
        all_genes.append(f'GENE_{i:04d}')
    
    n_genes = len(all_genes)
    
    # Calculate total cells
    total_cells = sum(info['n_cells'] for info in cell_types.values())
    
    print(f"Creating synthetic PBMC data:")
    print(f"  Cells: {total_cells}")
    print(f"  Genes: {n_genes}")
    print(f"  Cell types: {list(cell_types.keys())}")
    
    # Initialize expression matrix
    X = np.zeros((total_cells, n_genes))
    cell_labels = []
    cell_ids = []
    
    cell_idx = 0
    
    for cell_type, info in cell_types.items():
        n_cells = info['n_cells']
        marker_genes = info['marker_genes']
        
        print(f"  Generating {n_cells} {cell_type} cells...")
        
        for i in range(n_cells):
            cell_id = f"{cell_type}_{i:04d}"
            cell_ids.append(cell_id)
            cell_labels.append(cell_type)
            
            # Base expression (housekeeping genes)
            for gene in housekeeping_genes:
                if gene in all_genes:
                    gene_idx = all_genes.index(gene)
                    X[cell_idx, gene_idx] = np.random.poisson(50)  # Moderate expression
            
            # Marker gene expression (high for this cell type)
            for gene in marker_genes:
                if gene in all_genes:
                    gene_idx = all_genes.index(gene)
                    X[cell_idx, gene_idx] = np.random.poisson(200)  # High expression
            
            # Random background expression
            n_expressed = np.random.randint(100, 300)  # Number of genes expressed
            expressed_genes = np.random.choice(n_genes, n_expressed, replace=False)
            
            for gene_idx in expressed_genes:
                if X[cell_idx, gene_idx] == 0:  # Don't overwrite marker/housekeeping
                    X[cell_idx, gene_idx] = np.random.poisson(10)  # Low background
            
            cell_idx += 1
    
    # Create AnnData object
    adata = ad.AnnData(X=X)
    adata.obs_names = cell_ids
    adata.var_names = all_genes
    
    # Add metadata
    adata.obs['cell_type'] = cell_labels
    adata.obs['n_genes'] = (X > 0).sum(axis=1)
    adata.obs['total_counts'] = X.sum(axis=1)
    
    # Add gene metadata
    adata.var['gene_ids'] = all_genes
    adata.var['n_cells'] = (X > 0).sum(axis=0)
    
    # Create output directory
    os.makedirs('data', exist_ok=True)
    
    # Save the data
    output_path = 'data/pbmc_demo.h5ad'
    adata.write_h5ad(output_path)
    
    print(f"Saved demo PBMC data to {output_path}")
    print(f"Data shape: {adata.shape}")
    print("Cell type distribution:")
    print(adata.obs['cell_type'].value_counts())
    
    return adata

if __name__ == "__main__":
    create_demo_pbmc_data()
