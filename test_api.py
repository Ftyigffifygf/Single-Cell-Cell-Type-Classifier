import requests
import json

# Test the API endpoints
base_url = 'http://localhost:8000'

print('=== Testing Single-Cell Cell-Type Classifier API ===')

# Test health check
print('\\n1. Health Check:')
response = requests.get(f'{base_url}/')
print(f'Status: {response.status_code}')
print(f'Response: {response.json()}')

# Test model info
print('\\n2. Model Info:')
response = requests.get(f'{base_url}/info')
print(f'Status: {response.status_code}')
info = response.json()
print(f'Feature Dimension: {info[\"feature_dimension\"]}')
print(f'Classes: {info[\"classes\"]}')
print(f'Model Type: {info[\"model_type\"]}')

# Test prediction with T-cell markers
print('\\n3. Prediction Test (T-cell markers):')
test_data = {
    'gene_expression': [100.0, 0.0, 0.0, 80.0, 50.0],
    'gene_names': ['CD3D', 'CD19', 'CD14', 'CD8A', 'CD4']
}

response = requests.post(
    f'{base_url}/predict',
    headers={'Content-Type': 'application/json'},
    json=test_data
)

print(f'Status: {response.status_code}')
if response.status_code == 200:
    result = response.json()
    print(f'Predicted Cell Type: {result[\"cell_type\"]}')
    print(f'Confidence: {result[\"confidence\"]:.3f}')
    print('Top Probabilities:')
    sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
    for cell_type, prob in sorted_probs[:3]:
        print(f'  {cell_type}: {prob:.3f}')
else:
    print(f'Error: {response.text}')

print('\\n=== API Test Complete ===')
