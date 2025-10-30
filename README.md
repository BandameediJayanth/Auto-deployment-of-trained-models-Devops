# Auto-deployment-of-trained-models-Devops


# ML Model CLI Pipeline

This repository contains a CLI-first MLOps pipeline for training, validating, and making predictions with machine learning models. The tools are intentionally simple and automation-friendly so they can be used directly in CI pipelines or integrated into larger systems.

## Overview

The project focuses on three scripts:

- `src/train_model.py` — train a model, save model and metadata
- `src/validate_model.py` — validate a saved model and generate a report
- `src/predict.py` — make predictions from a saved model

All scripts are command-line driven. There is no server or API code included by design.

## Quick start

### Prerequisites

- Python 3.8 or newer
- Git

### Set up the project (Windows PowerShell)

```powershell
cd "C:\Users\banda\OneDrive\Desktop\Devops_Project"
.\setup.ps1
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Set up the project (Linux / macOS)

```bash
./setup.sh
source venv/bin/activate
pip install -r requirements.txt
```

## CLI workflow

### Train a model (non-interactive defaults available; interactive prompts when run in a TTY):

```bash
python src/train_model.py --model-name my_model --version 1.0.0
```

When run interactively, `train_model.py` will list existing models (if any) and allow selecting one to overwrite or entering a new model name and version.

### Validate a model (point to any `.pkl` model file):

```bash
python src/validate_model.py --model models/my_model_v1.0.0.pkl
```

### Predict using a model (features are space-separated floats):

```bash
python src/predict.py --model models/my_model_v1.0.0.pkl --features 1.0 2.0 3.0 4.0 5.0
```

### Run tests:

```bash
pytest
```

## Project layout

```
Devops_Project/
├── src/
│   ├── train_model.py        # training script (CLI)
│   ├── validate_model.py     # validation script (CLI)
│   ├── predict.py            # prediction script (CLI)
│   └── utils.py              # helper functions
├── models/                   # saved models and metadata
├── data/                     # datasets
├── tests/                    # pytest test suite
├── requirements.txt
├── README.md
└── PROJECT_SUMMARY.md
```

## How it works — from scratch

1. Training

	- `train_model.py` attempts to load `data/dataset.csv`. If the file is missing the script generates a synthetic dataset for experimentation.
	- The script trains a scikit-learn `RandomForestClassifier` on the data, computes metrics (accuracy, precision, recall, f1), and writes:
	  - `models/<model_name>_v<version>.pkl` — model saved with `joblib`
	  - `models/<model_name>_v<version>_metadata.json` — metadata containing `feature_names`, metrics, and version
	  - `models/latest_model.json` — pointer to the latest model and metadata

2. Validation

	- `validate_model.py --model <path>` loads a saved model and its metadata, loads test data from `data/dataset.csv`, and computes:
	  - performance metrics on the test set
	  - k-fold cross-validation scores
	  - prediction capability checks (single-sample prediction, predict_proba if available)
	- Outputs a JSON report `models/validation_report_<timestamp>.json` and prints a concise PASS/FAIL summary to the terminal.

3. Prediction

	- `predict.py --model <path> --features <f1> <f2> ...` loads the model and metadata, checks feature count, and prints a JSON prediction containing the class, probabilities (if available), and model version.

## Design choices and rationale

- CLI-first: Scripts are automation-friendly and easy to run in CI without requiring an API server.
- Explicit artifacts: model and metadata are persisted, and `latest_model.json` points to the current artifact for convenience.
- Test coverage: Unit and integration tests (pytest) ensure the core pipeline works and are used as a safety net for changes.

## Extensibility

- To use your own training code, replace the training steps in `src/train_model.py` but keep the metadata structure (especially `feature_names` and `metrics`).
- To add a serving layer, create a separate service that loads models from `models/` and calls `predict.py` logic or directly loads the `.pkl` via `joblib`.
- Integrate the CLI commands into CI pipelines to run training/validation and gate deployments on validation results.

## Removed / obsolete pieces

- API/server code, monitoring stacks, and Docker containerization instructions were intentionally removed to focus on a simple CLI-first workflow. These can be re-introduced as separate modules if needed.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/...`)
3. Run tests locally and ensure they pass
4. Open a PR with a clear description of the change

## Support & next steps

If you want help adapting this to CI, adding containers, or adding a serving API, tell me the target environment (Kubernetes, ECS, plain VM) and I will draft the necessary changes.
>>>>>>> 5ac396a (First push of Devops Project)
