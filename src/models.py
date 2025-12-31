"""
Machine Learning Models Module
Implements multiple ML models for sentiment classification.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, List
import logging
import joblib
from abc import ABC, abstractmethod

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

import sys
sys.path.append('..')
from config.settings import MODEL_CONFIG, HYPERPARAMETERS, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for sentiment analysis models."""
    
    def __init__(self, name: str):
        """
        Initialize base model.
        
        Args:
            name: Model name
        """
        self.name = name
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
    @abstractmethod
    def build_model(self, **kwargs):
        """Build the model architecture."""
        pass
    
    def fit(self, X_train, y_train):
        """
        Fit the model on training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info(f"Training {self.name}...")
        
        # Encode labels if they're strings
        if y_train.dtype == 'object':
            y_train = self.label_encoder.fit_transform(y_train)
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        logger.info(f"{self.name} training completed")
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = self.model.predict(X)
        
        # Decode labels if encoder was used
        if hasattr(self.label_encoder, 'classes_'):
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability estimates
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            logger.warning(f"{self.name} does not support probability prediction")
            return None
    
    def cross_validate(self, X, y, cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Labels
            cv: Number of folds
            
        Returns:
            Dictionary with mean and std of scores
        """
        logger.info(f"Performing {cv}-fold cross-validation for {self.name}...")
        
        if y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
        
        scores = cross_val_score(
            self.model, X, y, 
            cv=cv, 
            scoring='accuracy',
            n_jobs=MODEL_CONFIG['n_jobs']
        )
        
        result = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'all_scores': scores
        }
        
        logger.info(f"CV Score: {result['mean_score']:.4f} (+/- {result['std_score']:.4f})")
        
        return result
    
    def save(self, filename: str = None):
        """Save model to disk."""
        if filename is None:
            filename = f"{self.name.lower().replace(' ', '_')}.pkl"
        
        filepath = MODELS_DIR / filename
        joblib.dump({
            'model': self.model,
            'label_encoder': self.label_encoder,
            'name': self.name
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filename: str = None):
        """Load model from disk."""
        if filename is None:
            filename = f"{self.name.lower().replace(' ', '_')}.pkl"
        
        filepath = MODELS_DIR / filename
        data = joblib.load(filepath)
        
        self.model = data['model']
        self.label_encoder = data['label_encoder']
        self.name = data['name']
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")


class LogisticRegressionModel(BaseModel):
    """Logistic Regression classifier."""
    
    def __init__(self):
        super().__init__("Logistic Regression")
        self.build_model()
    
    def build_model(self, **kwargs):
        """Build logistic regression model."""
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=MODEL_CONFIG['random_state'],
            n_jobs=MODEL_CONFIG['n_jobs'],
            **kwargs
        )


class RandomForestModel(BaseModel):
    """Random Forest classifier."""
    
    def __init__(self):
        super().__init__("Random Forest")
        self.build_model()
    
    def build_model(self, **kwargs):
        """Build random forest model."""
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=MODEL_CONFIG['random_state'],
            n_jobs=MODEL_CONFIG['n_jobs'],
            **kwargs
        )


class SVMModel(BaseModel):
    """Support Vector Machine classifier."""
    
    def __init__(self):
        super().__init__("Support Vector Machine")
        self.build_model()
    
    def build_model(self, **kwargs):
        """Build SVM model."""
        self.model = SVC(
            kernel='linear',
            probability=True,  # Enable probability estimates
            random_state=MODEL_CONFIG['random_state'],
            **kwargs
        )


class NaiveBayesModel(BaseModel):
    """Naive Bayes classifier."""
    
    def __init__(self, variant: str = 'multinomial'):
        """
        Initialize Naive Bayes model.
        
        Args:
            variant: 'multinomial' or 'gaussian'
        """
        super().__init__(f"Naive Bayes ({variant})")
        self.variant = variant
        self.build_model()
    
    def build_model(self, **kwargs):
        """Build Naive Bayes model."""
        if self.variant == 'multinomial':
            self.model = MultinomialNB(**kwargs)
        else:
            self.model = GaussianNB(**kwargs)


class GradientBoostingModel(BaseModel):
    """Gradient Boosting classifier."""
    
    def __init__(self):
        super().__init__("Gradient Boosting")
        self.build_model()
    
    def build_model(self, **kwargs):
        """Build gradient boosting model."""
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            random_state=MODEL_CONFIG['random_state'],
            **kwargs
        )


class ModelTrainer:
    """Handles training and hyperparameter tuning for multiple models."""
    
    def __init__(self):
        """Initialize model trainer."""
        self.models = {}
        self.best_params = {}
        self.cv_scores = {}
    
    def add_model(self, model: BaseModel):
        """
        Add a model to the trainer.
        
        Args:
            model: Model instance
        """
        self.models[model.name] = model
        logger.info(f"Added model: {model.name}")
    
    def train_all(self, X_train, y_train):
        """
        Train all registered models.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info(f"Training {len(self.models)} models...")
        
        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                logger.info(f"✓ {name} trained successfully")
            except Exception as e:
                logger.error(f"✗ Failed to train {name}: {e}")
    
    def tune_hyperparameters(self, 
                            model: BaseModel,
                            X_train, 
                            y_train,
                            param_grid: Dict = None) -> Dict:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            model: Model to tune
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid for search
            
        Returns:
            Best parameters found
        """
        logger.info(f"Tuning hyperparameters for {model.name}...")
        
        if param_grid is None:
            model_key = model.name.lower().replace(' ', '_')
            param_grid = HYPERPARAMETERS.get(model_key, {})
        
        if not param_grid:
            logger.warning(f"No parameter grid provided for {model.name}")
            return {}
        
        # Encode labels if needed
        if y_train.dtype == 'object':
            y_train = model.label_encoder.fit_transform(y_train)
        
        grid_search = GridSearchCV(
            model.model,
            param_grid,
            cv=MODEL_CONFIG['cv_folds'],
            scoring='accuracy',
            n_jobs=MODEL_CONFIG['n_jobs'],
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best CV score: {best_score:.4f}")
        
        # Update model with best parameters
        model.model = grid_search.best_estimator_
        model.is_fitted = True
        
        self.best_params[model.name] = best_params
        
        return best_params
    
    def compare_models(self, X_train, y_train, cv: int = 5) -> pd.DataFrame:
        """
        Compare all models using cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv: Number of folds
            
        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Comparing {len(self.models)} models...")
        
        results = []
        
        for name, model in self.models.items():
            try:
                cv_result = model.cross_validate(X_train, y_train, cv=cv)
                results.append({
                    'Model': name,
                    'Mean CV Score': cv_result['mean_score'],
                    'Std CV Score': cv_result['std_score']
                })
                self.cv_scores[name] = cv_result
            except Exception as e:
                logger.error(f"Error comparing {name}: {e}")
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('Mean CV Score', ascending=False)
        
        return comparison_df
    
    def get_best_model(self) -> Tuple[str, BaseModel]:
        """
        Get the best performing model based on CV scores.
        
        Returns:
            Tuple of (model_name, model_instance)
        """
        if not self.cv_scores:
            raise ValueError("Run compare_models first")
        
        best_name = max(self.cv_scores, key=lambda k: self.cv_scores[k]['mean_score'])
        best_model = self.models[best_name]
        
        logger.info(f"Best model: {best_name}")
        
        return best_name, best_model


def handle_imbalanced_data(X_train, y_train) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle imbalanced datasets using SMOTE.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Resampled X_train and y_train
    """
    from collections import Counter
    
    logger.info(f"Original class distribution: {Counter(y_train)}")
    
    # Convert sparse matrix to dense for SMOTE
    if hasattr(X_train, 'toarray'):
        X_train = X_train.toarray()
    
    smote = SMOTE(random_state=MODEL_CONFIG['random_state'])
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    logger.info(f"Resampled class distribution: {Counter(y_resampled)}")
    
    return X_resampled, y_resampled


def main():
    """Example usage of model training."""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, 
                              n_informative=15, n_classes=3,
                              random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Add models
    trainer.add_model(LogisticRegressionModel())
    trainer.add_model(RandomForestModel())
    trainer.add_model(SVMModel())
    trainer.add_model(NaiveBayesModel(variant='gaussian'))
    
    # Compare models
    comparison = trainer.compare_models(X_train, y_train)
    print("\n=== Model Comparison ===")
    print(comparison)
    
    # Get best model
    best_name, best_model = trainer.get_best_model()
    print(f"\nBest Model: {best_name}")


if __name__ == "__main__":
    main()