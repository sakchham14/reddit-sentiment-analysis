"""
Model Evaluation Module
Comprehensive evaluation metrics and analysis tools.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('..')
from config.settings import VIZ_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and analysis."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.metrics = {}
        self.predictions = {}
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette(VIZ_CONFIG['color_palette'])
    
    def evaluate(self, 
                y_true, 
                y_pred, 
                y_proba=None,
                model_name: str = "Model") -> Dict:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            model_name: Name of the model
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        unique_labels = np.unique(y_true)
        for i, label in enumerate(unique_labels):
            metrics[f'precision_class_{label}'] = precision_per_class[i]
            metrics[f'recall_class_{label}'] = recall_per_class[i]
            metrics[f'f1_class_{label}'] = f1_per_class[i]
        
        # ROC AUC for binary/multiclass
        if y_proba is not None:
            try:
                if len(unique_labels) == 2:
                    # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    # Multiclass classification
                    metrics['roc_auc_ovr'] = roc_auc_score(
                        y_true, y_proba, 
                        multi_class='ovr', 
                        average='macro'
                    )
            except Exception as e:
                logger.warning(f"Could not compute ROC AUC: {e}")
        
        # Store results
        self.metrics[model_name] = metrics
        self.predictions[model_name] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        return metrics
    
    def print_metrics(self, model_name: str = None):
        """
        Print evaluation metrics in a formatted way.
        
        Args:
            model_name: Name of model to print (None for all)
        """
        if model_name:
            models = [model_name]
        else:
            models = list(self.metrics.keys())
        
        for name in models:
            if name not in self.metrics:
                logger.warning(f"No metrics found for {name}")
                continue
            
            print(f"\n{'='*60}")
            print(f"  {name} - Evaluation Metrics")
            print(f"{'='*60}")
            
            metrics = self.metrics[name]
            
            print(f"\nOverall Metrics:")
            print(f"  Accuracy:           {metrics['accuracy']:.4f}")
            print(f"  Precision (macro):  {metrics['precision_macro']:.4f}")
            print(f"  Recall (macro):     {metrics['recall_macro']:.4f}")
            print(f"  F1 Score (macro):   {metrics['f1_macro']:.4f}")
            
            if 'roc_auc' in metrics:
                print(f"  ROC AUC:            {metrics['roc_auc']:.4f}")
            elif 'roc_auc_ovr' in metrics:
                print(f"  ROC AUC (OvR):      {metrics['roc_auc_ovr']:.4f}")
            
            # Per-class metrics
            class_metrics = {k: v for k, v in metrics.items() 
                           if k.startswith(('precision_class_', 'recall_class_', 'f1_class_'))}
            
            if class_metrics:
                print(f"\nPer-Class Metrics:")
                classes = sorted(set(k.split('_')[-1] for k in class_metrics.keys()))
                
                for cls in classes:
                    print(f"\n  Class {cls}:")
                    print(f"    Precision: {metrics.get(f'precision_class_{cls}', 0):.4f}")
                    print(f"    Recall:    {metrics.get(f'recall_class_{cls}', 0):.4f}")
                    print(f"    F1 Score:  {metrics.get(f'f1_class_{cls}', 0):.4f}")
    
    def get_classification_report(self, model_name: str) -> str:
        """
        Get detailed classification report.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Classification report string
        """
        if model_name not in self.predictions:
            raise ValueError(f"No predictions found for {model_name}")
        
        preds = self.predictions[model_name]
        return classification_report(preds['y_true'], preds['y_pred'])
    
    def plot_confusion_matrix(self, 
                             model_name: str,
                             normalize: bool = False,
                             figsize: Tuple = None):
        """
        Plot confusion matrix.
        
        Args:
            model_name: Name of the model
            normalize: Whether to normalize values
            figsize: Figure size
        """
        if model_name not in self.predictions:
            raise ValueError(f"No predictions found for {model_name}")
        
        preds = self.predictions[model_name]
        cm = confusion_matrix(preds['y_true'], preds['y_pred'])
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = f'Normalized Confusion Matrix - {model_name}'
        else:
            fmt = 'd'
            title = f'Confusion Matrix - {model_name}'
        
        figsize = figsize or VIZ_CONFIG['figure_size']
        plt.figure(figsize=figsize)
        
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   cbar=True, square=True)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_roc_curve(self, model_name: str, figsize: Tuple = None):
        """
        Plot ROC curve for binary classification.
        
        Args:
            model_name: Name of the model
            figsize: Figure size
        """
        if model_name not in self.predictions:
            raise ValueError(f"No predictions found for {model_name}")
        
        preds = self.predictions[model_name]
        
        if preds['y_proba'] is None:
            logger.warning(f"{model_name} does not have probability predictions")
            return
        
        y_true = preds['y_true']
        y_proba = preds['y_proba']
        
        # Binary classification
        if y_proba.shape[1] == 2:
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = roc_auc_score(y_true, y_proba[:, 1])
            
            figsize = figsize or VIZ_CONFIG['figure_size']
            plt.figure(figsize=figsize)
            
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                    label='Random Classifier')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            return plt.gcf()
        else:
            logger.warning("ROC curve plotting for multiclass not implemented")
    
    def plot_precision_recall_curve(self, model_name: str, figsize: Tuple = None):
        """
        Plot precision-recall curve for binary classification.
        
        Args:
            model_name: Name of the model
            figsize: Figure size
        """
        if model_name not in self.predictions:
            raise ValueError(f"No predictions found for {model_name}")
        
        preds = self.predictions[model_name]
        
        if preds['y_proba'] is None:
            logger.warning(f"{model_name} does not have probability predictions")
            return
        
        y_true = preds['y_true']
        y_proba = preds['y_proba']
        
        # Binary classification
        if y_proba.shape[1] == 2:
            precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
            avg_precision = average_precision_score(y_true, y_proba[:, 1])
            
            figsize = figsize or VIZ_CONFIG['figure_size']
            plt.figure(figsize=figsize)
            
            plt.plot(recall, precision, color='blue', lw=2,
                    label=f'PR curve (AP = {avg_precision:.2f})')
            plt.axhline(y=y_true.mean(), color='red', linestyle='--',
                       label=f'Baseline (AP = {y_true.mean():.2f})')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            plt.title(f'Precision-Recall Curve - {model_name}', 
                     fontsize=14, fontweight='bold')
            plt.legend(loc="lower left")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            return plt.gcf()
    
    def compare_models(self, metric: str = 'f1_macro') -> pd.DataFrame:
        """
        Compare multiple models on a specific metric.
        
        Args:
            metric: Metric to compare
            
        Returns:
            DataFrame with comparison
        """
        if not self.metrics:
            raise ValueError("No model metrics available")
        
        comparison = []
        for name, metrics in self.metrics.items():
            if metric in metrics:
                comparison.append({
                    'Model': name,
                    metric: metrics[metric]
                })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values(metric, ascending=False)
        
        return df
    
    def plot_model_comparison(self, 
                             metrics: List[str] = None,
                             figsize: Tuple = None):
        """
        Plot comparison of multiple models across metrics.
        
        Args:
            metrics: List of metrics to compare
            figsize: Figure size
        """
        if metrics is None:
            metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        # Collect data
        data = []
        for name, model_metrics in self.metrics.items():
            for metric in metrics:
                if metric in model_metrics:
                    data.append({
                        'Model': name,
                        'Metric': metric,
                        'Score': model_metrics[metric]
                    })
        
        df = pd.DataFrame(data)
        
        figsize = figsize or (12, 6)
        plt.figure(figsize=figsize)
        
        # Create grouped bar plot
        models = df['Model'].unique()
        x = np.arange(len(metrics))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            model_data = df[df['Model'] == model]
            scores = [model_data[model_data['Metric'] == m]['Score'].values[0] 
                     if len(model_data[model_data['Metric'] == m]) > 0 else 0
                     for m in metrics]
            
            plt.bar(x + i * width, scores, width, label=model)
        
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Model Comparison Across Metrics', fontsize=14, fontweight='bold')
        plt.xticks(x + width * (len(models) - 1) / 2, metrics, rotation=45)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def save_results(self, filename: str = 'evaluation_results.csv'):
        """
        Save all evaluation results to CSV.
        
        Args:
            filename: Output filename
        """
        from config.settings import PROCESSED_DATA_DIR
        
        # Convert metrics to DataFrame
        rows = []
        for model_name, metrics in self.metrics.items():
            row = {'Model': model_name}
            row.update(metrics)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        filepath = PROCESSED_DATA_DIR / filename
        df.to_csv(filepath, index=False)
        
        logger.info(f"Evaluation results saved to {filepath}")


def main():
    """Example usage of ModelEvaluator."""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_classes=3, 
                              n_informative=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Evaluate
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_test, y_pred, y_proba, "Random Forest")
    
    evaluator.print_metrics()
    
    print("\n=== Classification Report ===")
    print(evaluator.get_classification_report("Random Forest"))


if __name__ == "__main__":
    main()