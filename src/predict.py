
import argparse
import json
import sys
import os
import joblib
import numpy as np

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/ml_classifier_v1.0.0.pkl')
DEFAULT_METADATA_PATH = os.path.join(os.path.dirname(__file__), '../models/ml_classifier_v1.0.0_metadata.json')

def load_model(model_path):
    metadata_path = os.path.splitext(model_path)[0] + '_metadata.json'
    model = joblib.load(model_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return model, metadata

def main():
    parser = argparse.ArgumentParser(description='Make a prediction using the trained model.')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH, help='Path to the trained model file (.pkl)')
    parser.add_argument('--features', nargs='+', type=float, required=True, help='Feature values as space-separated numbers')
    args = parser.parse_args()

    features = np.array(args.features, dtype=float).reshape(1, -1)
    try:
        model, metadata = load_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)
    expected = len(metadata['feature_names'])
    if features.shape[1] != expected:
        print(f"Error: Expected {expected} features, got {features.shape[1]}", file=sys.stderr)
        sys.exit(1)
    try:
        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0].tolist() if hasattr(model, 'predict_proba') else []
        result = {
            'prediction': int(pred),
            'probability': proba,
            'model_version': metadata['version']
        }
        print("Prediction:", json.dumps(result))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == '__main__':
    main()
