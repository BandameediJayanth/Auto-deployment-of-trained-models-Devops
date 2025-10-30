"""
Custom Model Training - Use Your Own Data
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import json
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('custom_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CustomModelTrainer:
    def __init__(self, model_name="custom_classifier", model_version="1.0.0"):
        self.model_name = model_name
        self.model_version = model_version
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
    def load_csv_data(self, csv_file_path, target_column):
        """
        Load data from CSV file
        
        Args:
            csv_file_path (str): Path to your CSV file
            target_column (str): Name of the column you want to predict
        """
        logger.info(f"Loading data from {csv_file_path}")
        
        # Load CSV file
        df = pd.read_csv(csv_file_path)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        
        # Check if target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in CSV. Available columns: {list(df.columns)}")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Store feature names
        self.feature_names = list(X.columns)
        logger.info(f"Features: {self.feature_names}")
        logger.info(f"Target classes: {sorted(y.unique())}")
        
        return X, y
    
    def load_custom_data(self, X_data, y_data, feature_names=None):
        """
        Load data from numpy arrays or pandas dataframes
        
        Args:
            X_data: Feature data (numpy array or pandas DataFrame)
            y_data: Target data (numpy array or pandas Series)
            feature_names: List of feature names (optional)
        """
        logger.info("Loading custom data")
        
        # Convert to pandas if needed
        if not isinstance(X_data, pd.DataFrame):
            X_data = pd.DataFrame(X_data)
        if not isinstance(y_data, pd.Series):
            y_data = pd.Series(y_data)
        
        # Set feature names
        if feature_names:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_data.shape[1])]
            X_data.columns = self.feature_names
        
        logger.info(f"Data shape: {X_data.shape}")
        logger.info(f"Features: {self.feature_names}")
        logger.info(f"Target classes: {sorted(y_data.unique())}")
        
        return X_data, y_data
    
    def preprocess_data(self, X, y):
        """
        Preprocess the data (handle missing values, encode categories, scale features)
        """
        logger.info("Preprocessing data...")
        
        # Handle missing values
        if X.isnull().sum().sum() > 0:
            logger.warning("Found missing values, filling with median/mode")
            # Fill numeric columns with median
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
            
            # Fill categorical columns with mode
            categorical_cols = X.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                X[col] = X[col].fillna(X[col].mode()[0])
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            logger.info(f"Encoding categorical columns: {list(categorical_cols)}")
            # Simple label encoding for categorical features
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        # Encode target if it's categorical
        if y.dtype == 'object' or not np.issubdtype(y.dtype, np.number):
            logger.info("Encoding target variable")
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y.values
        
        logger.info("Preprocessing completed")
        return X_scaled, y_encoded
    
    def train_model(self, X, y, model_params=None):
        """
        Train the model
        
        Args:
            X: Features
            y: Target
            model_params: Custom parameters for the model
        """
        logger.info("Starting model training...")
        
        # Default model parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        
        # Use custom parameters if provided (only valid ones)
        if model_params:
            valid_params = {}
            for key, value in model_params.items():
                if key in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'random_state']:
                    valid_params[key] = value
            default_params.update(valid_params)
        
        # Create and train model
        self.model = RandomForestClassifier(**default_params)
        self.model.fit(X, y)
        
        logger.info("Model training completed")
        return self.model
    
    def evaluate_model(self, X, y):
        """
        Evaluate model performance
        """
        logger.info("Evaluating model...")
        
        # Initialize model if not already done
        if self.model is None:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train on training set
        self.model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': len(self.feature_names),
            'training_date': datetime.now().isoformat(),
            'model_version': self.model_version
        }
        
        logger.info(f"Model Performance:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"  Precision: {metrics['precision']:.3f}")
        logger.info(f"  Recall: {metrics['recall']:.3f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.3f}")
        
        return metrics
    
    def save_model(self, metrics):
        """
        Save the trained model and metadata
        """
        # Save model
        model_filename = f"{self.model_name}_v{self.model_version}.pkl"
        model_path = self.models_dir / model_filename
        
        # Create model package (includes scaler and label encoder)
        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder if hasattr(self.label_encoder, 'classes_') else None,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_package, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'version': self.model_version,
            'model_path': str(model_path),
            'feature_names': self.feature_names,
            'metrics': metrics,
            'model_type': 'RandomForestClassifier',
            'sklearn_version': '1.4.2'
        }
        
        metadata_filename = f"{self.model_name}_v{self.model_version}_metadata.json"
        metadata_path = self.models_dir / metadata_filename
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
        return model_path, metadata_path

def train_from_csv():
    """
    Example: Train model from CSV file
    """
    print("🎯 Training Model from CSV File")
    print("=" * 50)
    
    # Get CSV file path from user
    csv_path = input("Enter path to your CSV file: ").strip()
    if not csv_path:
        print("❌ No file path provided")
        return
    
    # Check if file exists
    if not Path(csv_path).exists():
        print(f"❌ File not found: {csv_path}")
        return
    
    # Get target column name
    target_col = input("Enter the name of the column you want to predict: ").strip()
    if not target_col:
        print("❌ No target column provided")
        return
    
    # Get model name
    model_name = input("Enter model name (default: my_custom_model): ").strip()
    if not model_name:
        model_name = "my_custom_model"
    
    try:
        # Initialize trainer
        trainer = CustomModelTrainer(model_name=model_name)
        
        # Load data
        X, y = trainer.load_csv_data(csv_path, target_col)
        
        # Preprocess data
        X_processed, y_processed = trainer.preprocess_data(X, y)
        
        # Train and evaluate
        metrics = trainer.evaluate_model(X_processed, y_processed)
        
        # Save model
        model_path, metadata_path = trainer.save_model(metrics)
        
        print("\n🎉 Training Complete!")
        print(f"✅ Model saved: {model_path}")
        print(f"📊 Accuracy: {metrics['accuracy']:.3f}")
        print(f"🎯 Ready for deployment!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def train_from_arrays():
    """
    Example: Train model from numpy arrays (for advanced users)
    """
    print("🎯 Training Model from Arrays")
    print("=" * 50)
    
    print("This example creates sample data. Replace with your own arrays.")
    
    # Example: Create sample data (replace this with your data)
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Create features
    X = np.random.randn(n_samples, n_features)
    
    # Create target (example: based on sum of first 3 features)
    y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(int)
    
    # Feature names
    feature_names = [f"custom_feature_{i}" for i in range(n_features)]
    
    try:
        # Initialize trainer
        trainer = CustomModelTrainer(model_name="array_model")
        
        # Load data
        X_df, y_series = trainer.load_custom_data(X, y, feature_names)
        
        # Preprocess data
        X_processed, y_processed = trainer.preprocess_data(X_df, y_series)
        
        # Train and evaluate
        metrics = trainer.evaluate_model(X_processed, y_processed)
        
        # Save model
        model_path, metadata_path = trainer.save_model(metrics)
        
        print("\n🎉 Training Complete!")
        print(f"✅ Model saved: {model_path}")
        print(f"📊 Accuracy: {metrics['accuracy']:.3f}")
        print(f"🎯 Ready for deployment!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """
    Main function - choose training method
    """
    print("🚀 Custom Model Training")
    print("=" * 40)
    print("Choose your training method:")
    print("1. Train from CSV file")
    print("2. Train from arrays (advanced)")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        train_from_csv()
    elif choice == "2":
        train_from_arrays()
    elif choice == "3":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
