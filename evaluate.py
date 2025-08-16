import numpy as np
import toml
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path

def load_config():
    with open('config.toml', 'r') as f:
        return toml.load(f)

def evaluate_model(config):
    embeddings_path = config['data']['embeddings_npz']
    print(f'Loading embeddings from {embeddings_path}')
    data = np.load(embeddings_path, allow_pickle=True)
    
    embeddings = data['embeddings']
    labels = data['labels']
    
    # Load model bundle
    model_path = config['training']['model_path']
    print(f'Loading model from {model_path}')
    model_bundle = joblib.load(model_path)
    
    scaler = model_bundle['scaler']
    label_encoder = model_bundle['label_encoder']
    classifier = model_bundle['classifier']
    
    # Encode labels
    encoded_labels = label_encoder.transform(labels)
    
    # Split data (same split as training)
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, encoded_labels,
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state'],
        stratify=encoded_labels
    )
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = classifier.predict(X_test_scaled)
    y_pred_proba = classifier.predict_proba(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Classification report
    report = classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create evaluation results
    evaluation_results = {
        'accuracy': float(accuracy),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'class_names': label_encoder.classes_.tolist(),
        'test_size': len(y_test),
        'feature_dimension': embeddings.shape[1]
    }
    
    # Print results
    print('Evaluation Results:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Macro F1-Score: {report[\"macro avg\"][\"f1-score\"]:.4f}')
    print(f'Weighted F1-Score: {report[\"weighted avg\"][\"f1-score\"]:.4f}')
    
    print('\\nPer-class results:')
    for class_name in label_encoder.classes_:
        if class_name in report:
            metrics = report[class_name]
            print(f'  {class_name}:')
            print(f'    Precision: {metrics[\"precision\"]:.4f}')
            print(f'    Recall: {metrics[\"recall\"]:.4f}')
            print(f'    F1-Score: {metrics[\"f1-score\"]:.4f}')
            print(f'    Support: {metrics[\"support\"]}')
    
    # Create output directory
    report_path = config['evaluation']['report_path']
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    # Save evaluation report
    with open(report_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f'\\nSaved evaluation report to {report_path}')
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    # Save confusion matrix plot
    cm_path = config['evaluation']['confusion_matrix_path']
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f'Saved confusion matrix plot to {cm_path}')
    
    # Identify top confusions
    print('\\nTop confusions:')
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    confusions = []
    for i in range(len(label_encoder.classes_)):
        for j in range(len(label_encoder.classes_)):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    'true_class': label_encoder.classes_[i],
                    'predicted_class': label_encoder.classes_[j],
                    'count': int(cm[i, j]),
                    'rate': float(cm_normalized[i, j])
                })
    
    # Sort by confusion rate
    confusions.sort(key=lambda x: x['rate'], reverse=True)
    
    for conf in confusions[:5]:  # Top 5 confusions
        print(f'  {conf[\"true_class\"]} -> {conf[\"predicted_class\"]}: {conf[\"count\"]} ({conf[\"rate\"]:.3f})')
    
    return evaluation_results

if __name__ == '__main__':
    config = load_config()
    results = evaluate_model(config)
