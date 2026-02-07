# 📥 Guide: Importing External Models (e.g., Kaggle)

To ensure that models from external sources (like Kaggle, HuggingFace, or other teams) are **Production Ready**, our system runs a rigorous ingestion pipeline.

## 🛑 Requirements
For a model to be accepted, it must be placed in the `input_models/` folder.

### 1. The Model File (Required)
- **Format:** `.pkl`, `.joblib`, or `.sav`
- **Compatibility:** Must be loadable via `joblib` and have a `.predict()` method.

### 2. Validation Data (Highly Recommended)
To prove the model actually works (and isn't just a random number generator), provide a `validation.csv` file.
- **Location:** `input_models/validation.csv`
- **Format:** CSV with headers.
    - **Features:** Columns matching the model's input.
    - **Target:** The *last column* is assumed to be the ground truth label.

### 3. Configuration (Optional)
You can define custom success criteria by providing a `config.json`.
- **Location:** `input_models/config.json`
- **Content:**
```json
{
  "min_accuracy": 0.85,
  "min_f1_score": 0.80,
  "max_latency_ms": 50.0
}
```

---

## 🚀 The Ingestion Process
When you run the deployment script with `-Ingest`:

1.  **🔍 Analysis**: The system scans the model file and checks for companion files (`validation.csv`, `config.json`).
2.  **🧪 Verification**:
    *   **Smoke Test:** Sends random noise to ensure the model doesn't crash.
    *   **Validation Test:** (If `validation.csv` exists) Runs predictions and calculates Accuracy/F1. Fails if below thresholds.
    *   **Latency Check:** Measures average inference time.
3.  **🏁 Verdict**: A report is generated in `reports/final_verdict.md`.
4.  **📦 Promotion**: If (and only if) the verdict is **READY**, the model is moved to `models/production/`.

## ❌ Common Failure Reasons
*   **Missing Libraries:** The environment running the pipeline must have the same libraries installed (e.g., `scikit-learn`, `xgboost`).
*   **Input Mismatch:** The model expects 12 features, but `validation.csv` has 10.
*   **Low Performance:** The model's accuracy on `validation.csv` is lower than `min_accuracy`.

## 📝 Example Workflow
1. Download `titanic_model.pkl` from Kaggle.
2. Prepare a small `validation.csv` (10-20 rows) from the test set.
3. Place both in `input_models/`.
4. Run:
   ```powershell
   powershell -ExecutionPolicy Bypass -File ci-cd/deploy.ps1 -Ingest
   ```
