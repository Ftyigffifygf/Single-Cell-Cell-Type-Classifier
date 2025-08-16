import numpy as np
import toml
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path

def load_config():
    with open('config.toml', 'r') as f:
        return toml.load(f)

def train_classifier(config):
    embeddings_path = config['data']['embeddings_npz']
    print(f'Loading embeddings from {embeddings_path}')
    data = np.load(embeddings_path, allow_pickle=True)
    
    embeddings = data['embeddings']
    labels = data['labels']
    cell_ids = data['cell_ids']
    
    print(f'Loaded embeddings shape: {embeddings.shape}')
    print(f'Number of labels: {len(labels)}')
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    print(f'Label classes: {label_encoder.classes_}')
    print('Label distribution:')
    unique, counts = np.unique(encoded_labels, return_counts=True)
    for i, (label_idx, count) in enumerate(zip(unique, counts)):
        print(f'  {label_encoder.classes_[label_idx]}: {count}')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, encoded_labels,
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state'],
        stratify=encoded_labels
    )
    
    print(f'Training set size: {X_train.shape[0]}')
    print(f'Test set size: {X_test.shape[0]}')
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train classifier
    classifier_type = config['training']['classifier_type']
    
    if classifier_type == 'mlp':
        print('Training MLP classifier...')
        classifier = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=500,
            random_state=config['training']['random_state'],
            early_stopping=True,
            validation_fraction=0.1
        )
    else:
        print('Training Logistic Regression classifier...')
        classifier = LogisticRegression(
            max_iter=1000,
            random_state=config['training']['random_state']
        )
    
    classifier.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    y_pred = classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f'Test accuracy: {accuracy:.4f}')
    
    # Detailed classification report
    report = classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        output_dict=True
    )
    
    print('Classification Report:')
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Create model bundle
    model_bundle = {
        'scaler': scaler,
        'label_encoder': label_encoder,
        'classifier': classifier,
        'feature_dim': embeddings.shape[1],
        'classes': label_encoder.classes_.tolist()
    }
    
    # Create output directory
    model_path = config['training']['model_path']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model bundle
    joblib.dump(model_bundle, model_path)
    print(f'Saved model bundle to {model_path}')
    
    return model_bundle, report

if __name__ == '__main__':
    config = load_config()
    model_bundle, report = train_classifier(config)
