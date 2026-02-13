"""
Generate Confusion Matrix and Performance Visualizations
For Research Paper Documentation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import joblib
import json
import os

# Set style
plt.style.use('ggplot')

def load_model_and_data():
    """Load the trained model and test data"""
    # Load latest model info
    latest_model_path = 'models/latest_model.json'
    
    # Check if latest_model.json exists
    if os.path.exists(latest_model_path):
        with open(latest_model_path, 'r') as f:
            latest_info = json.load(f)
        model_path = latest_info['model_path']
        metadata_path = latest_info['metadata_path']
    else:
        # Fallback to v1.0.0 if latest_model.json doesn't exist
        model_path = 'models/breast_cancer_model_v1.0.0.pkl'
        metadata_path = 'models/breast_cancer_model_v1.0.0_metadata.json'
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    
    # Load metadata
    print(f"Loading metadata from: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load data
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Split data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return model, X_test, y_test, metadata

def plot_confusion_matrix(y_true, y_pred, save_path='reports/confusion_matrix.png'):
    """Generate and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    
    # Create heatmap using imshow
    im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, label='Count')
    
    # Add labels
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Benign', 'Malignant'])
    plt.yticks(tick_marks, ['Benign', 'Malignant'])
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=20, fontweight='bold')
    
    plt.title('Confusion Matrix - Breast Cancer Prediction', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add accuracy text
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    plt.text(0.5, -0.15, f'Accuracy: {accuracy:.4f}', 
             ha='center', transform=plt.gca().transAxes, fontsize=11)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")
    plt.close()

def plot_roc_curve(y_true, y_proba, save_path='reports/roc_curve.png'):
    """Generate and save ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Breast Cancer Prediction', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC curve saved to {save_path}")
    plt.close()

def plot_precision_recall_curve(y_true, y_proba, save_path='reports/precision_recall_curve.png'):
    """Generate and save Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR curve (AUC = {pr_auc:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve - Breast Cancer Prediction', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Precision-Recall curve saved to {save_path}")
    plt.close()

def plot_feature_importance(model, feature_names, save_path='reports/feature_importance.png'):
    """Generate and save feature importance plot"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15 features
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], color='steelblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Feature importance saved to {save_path}")
    plt.close()

def generate_all_visualizations():
    """Generate all visualizations for research paper"""
    print("\n" + "="*60)
    print("GENERATING RESEARCH PAPER VISUALIZATIONS")
    print("="*60 + "\n")
    
    # Load model and data
    print("Loading model and data...")
    model, X_test, y_test, metadata = load_model_and_data()
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba)
    plot_precision_recall_curve(y_test, y_proba)
    plot_feature_importance(model, metadata['feature_names'])
    
    print("\n" + "="*60)
    print("✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
    print("="*60)
    print("\nFiles saved in 'reports/' directory:")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - precision_recall_curve.png")
    print("  - feature_importance.png")
    print("\nThese images can be used in your research paper.\n")

if __name__ == "__main__":
    generate_all_visualizations()
