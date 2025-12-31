"""
Main Execution Script for Reddit Sentiment Analysis
Orchestrates the complete pipeline from data collection to evaluation.
"""
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from config.settings import (
    COLLECTION_CONFIG, MODEL_CONFIG, RAW_DATA_DIR, 
    PROCESSED_DATA_DIR, MODELS_DIR
)
from src.data_collection import RedditDataCollector
from src.preprocessing import TextPreprocessor
from src.feature_extraction import (
    TFIDFFeatureExtractor, EmbeddingFeatureExtractor, 
    HybridFeatureExtractor
)
from src.models import (
    LogisticRegressionModel, RandomForestModel, SVMModel,
    NaiveBayesModel, ModelTrainer, handle_imbalanced_data
)
from src.evaluation import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SentimentAnalysisPipeline:
    """Complete pipeline for Reddit sentiment analysis."""
    
    def __init__(self):
        """Initialize pipeline components."""
        self.data_collector = None
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = None
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.features = None
        self.labels = None
        
        logger.info("Pipeline initialized")
    
    def collect_data(self, use_existing: bool = False):
        """
        Collect Reddit data or load existing data.
        
        Args:
            use_existing: Whether to use existing data files
        """
        logger.info("="*60)
        logger.info("STEP 1: Data Collection")
        logger.info("="*60)
        
        if use_existing:
            # Load most recent data files
            data_files = sorted(RAW_DATA_DIR.glob('posts_*.csv'))
            if data_files:
                latest_file = data_files[-1]
                logger.info(f"Loading existing data from {latest_file}")
                self.raw_data = pd.read_csv(latest_file)
                return
        
        # Collect new data
        self.data_collector = RedditDataCollector()
        posts_df, comments_df = self.data_collector.collect_dataset(
            subreddits=COLLECTION_CONFIG['subreddits'],
            post_limit=COLLECTION_CONFIG['post_limit'],
            comment_limit=COLLECTION_CONFIG['comment_limit'],
            time_filter=COLLECTION_CONFIG['time_filter']
        )
        
        # Combine posts and comments
        self.raw_data = pd.concat([
            posts_df[['title', 'text', 'score']].rename(columns={'text': 'content'}),
            comments_df[['text', 'score']].rename(columns={'text': 'content'})
        ], ignore_index=True)
        
        # Combine title and content for posts
        self.raw_data['full_text'] = (
            self.raw_data['title'].fillna('') + ' ' + 
            self.raw_data['content'].fillna('')
        ).str.strip()
        
        logger.info(f"Total samples collected: {len(self.raw_data)}")
    
    def label_data(self):
        """
        Create sentiment labels based on scores.
        This is a simple heuristic - you should use labeled data for production.
        """
        logger.info("Creating sentiment labels...")
        
        # Simple rule-based labeling (replace with actual labeled data)
        def score_to_sentiment(score):
            if score > 5:
                return 'positive'
            elif score < -1:
                return 'negative'
            else:
                return 'neutral'
        
        self.raw_data['sentiment'] = self.raw_data['score'].apply(score_to_sentiment)
        
        # Display distribution
        sentiment_counts = self.raw_data['sentiment'].value_counts()
        logger.info(f"Sentiment distribution:\n{sentiment_counts}")
        
        return self.raw_data
    
    def preprocess_data(self):
        """Preprocess text data."""
        logger.info("="*60)
        logger.info("STEP 2: Data Preprocessing")
        logger.info("="*60)
        
        # Preprocess text
        self.processed_data = self.preprocessor.preprocess_dataframe(
            self.raw_data,
            text_column='full_text',
            output_column='cleaned_text'
        )
        
        # Add text features
        self.processed_data = self.preprocessor.add_text_features(
            self.processed_data,
            text_column='full_text'
        )
        
        # Remove samples with empty cleaned text
        self.processed_data = self.processed_data[
            self.processed_data['cleaned_text'].str.strip() != ''
        ]
        
        logger.info(f"Samples after preprocessing: {len(self.processed_data)}")
        
        # Save processed data
        output_file = PROCESSED_DATA_DIR / 'processed_data.csv'
        self.processed_data.to_csv(output_file, index=False)
        logger.info(f"Processed data saved to {output_file}")
    
    def extract_features(self, feature_type: str = 'tfidf'):
        """
        Extract features from preprocessed text.
        
        Args:
            feature_type: 'tfidf', 'embeddings', or 'hybrid'
        """
        logger.info("="*60)
        logger.info("STEP 3: Feature Extraction")
        logger.info("="*60)
        
        texts = self.processed_data['cleaned_text'].tolist()
        self.labels = self.processed_data['sentiment'].values
        
        if feature_type == 'tfidf':
            logger.info("Extracting TF-IDF features...")
            self.feature_extractor = TFIDFFeatureExtractor()
            self.features = self.feature_extractor.fit_transform(texts)
            
        elif feature_type == 'embeddings':
            logger.info("Extracting embedding features...")
            self.feature_extractor = EmbeddingFeatureExtractor()
            self.features = self.feature_extractor.encode(texts)
            
        elif feature_type == 'hybrid':
            logger.info("Extracting hybrid features...")
            self.feature_extractor = HybridFeatureExtractor()
            self.feature_extractor.fit(texts)
            self.features = self.feature_extractor.transform(texts)
        
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        logger.info(f"Features shape: {self.features.shape}")
    
    def train_models(self, 
                    use_smote: bool = False,
                    tune_hyperparameters: bool = False):
        """
        Train multiple ML models.
        
        Args:
            use_smote: Whether to use SMOTE for class balancing
            tune_hyperparameters: Whether to perform hyperparameter tuning
        """
        logger.info("="*60)
        logger.info("STEP 4: Model Training")
        logger.info("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels,
            test_size=MODEL_CONFIG['test_size'],
            random_state=MODEL_CONFIG['random_state'],
            stratify=self.labels
        )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        # Handle imbalanced data
        if use_smote:
            logger.info("Applying SMOTE for class balancing...")
            X_train, y_train = handle_imbalanced_data(X_train, y_train)
        
        # Add models to trainer
        logger.info("Initializing models...")
        self.trainer.add_model(LogisticRegressionModel())
        self.trainer.add_model(RandomForestModel())
        self.trainer.add_model(SVMModel())
        
        # Use Multinomial NB for sparse features, Gaussian for dense
        if hasattr(X_train, 'toarray'):
            self.trainer.add_model(NaiveBayesModel(variant='multinomial'))
        else:
            self.trainer.add_model(NaiveBayesModel(variant='gaussian'))
        
        # Train all models
        self.trainer.train_all(X_train, y_train)
        
        # Hyperparameter tuning (optional, time-consuming)
        if tune_hyperparameters:
            logger.info("Performing hyperparameter tuning...")
            for name, model in self.trainer.models.items():
                try:
                    self.trainer.tune_hyperparameters(model, X_train, y_train)
                except Exception as e:
                    logger.warning(f"Tuning failed for {name}: {e}")
        
        # Store train/test splits for evaluation
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        
        return X_train, X_test, y_train, y_test
    
    def evaluate_models(self):
        """Evaluate all trained models."""
        logger.info("="*60)
        logger.info("STEP 5: Model Evaluation")
        logger.info("="*60)
        
        # Evaluate each model
        for name, model in self.trainer.models.items():
            try:
                # Make predictions
                y_pred = model.predict(self.X_test)
                y_proba = model.predict_proba(self.X_test)
                
                # Evaluate
                self.evaluator.evaluate(
                    self.y_test, y_pred, y_proba, name
                )
                
            except Exception as e:
                logger.error(f"Evaluation failed for {name}: {e}")
        
        # Print all metrics
        self.evaluator.print_metrics()
        
        # Save results
        self.evaluator.save_results()
    
    def visualize_results(self):
        """Create visualization plots."""
        logger.info("="*60)
        logger.info("STEP 6: Visualization")
        logger.info("="*60)
        
        # Model comparison
        fig = self.evaluator.plot_model_comparison()
        fig.savefig(PROCESSED_DATA_DIR / 'model_comparison.png', 
                   dpi=300, bbox_inches='tight')
        logger.info("Model comparison plot saved")
        
        # Confusion matrices for each model
        for name in self.trainer.models.keys():
            try:
                fig = self.evaluator.plot_confusion_matrix(name, normalize=True)
                filename = f'confusion_matrix_{name.lower().replace(" ", "_")}.png'
                fig.savefig(PROCESSED_DATA_DIR / filename, 
                           dpi=300, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Could not plot confusion matrix for {name}: {e}")
        
        logger.info("Visualization complete")
    
    def run_complete_pipeline(self, 
                             use_existing_data: bool = False,
                             feature_type: str = 'tfidf',
                             use_smote: bool = False,
                             tune_hyperparameters: bool = False):
        """
        Run the complete pipeline.
        
        Args:
            use_existing_data: Use existing data files
            feature_type: Type of features to extract
            use_smote: Use SMOTE for balancing
            tune_hyperparameters: Perform hyperparameter tuning
        """
        logger.info("\n" + "="*60)
        logger.info("STARTING REDDIT SENTIMENT ANALYSIS PIPELINE")
        logger.info("="*60 + "\n")
        
        try:
            # Step 1: Collect data
            self.collect_data(use_existing=use_existing_data)
            
            # Label data (or load pre-labeled data)
            self.label_data()
            
            # Step 2: Preprocess
            self.preprocess_data()
            
            # Step 3: Extract features
            self.extract_features(feature_type=feature_type)
            
            # Step 4: Train models
            self.train_models(
                use_smote=use_smote,
                tune_hyperparameters=tune_hyperparameters
            )
            
            # Step 5: Evaluate
            self.evaluate_models()
            
            # Step 6: Visualize
            self.visualize_results()
            
            logger.info("\n" + "="*60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*60 + "\n")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


def main():
    """Main execution function with CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Reddit Sentiment Analysis Pipeline'
    )
    
    parser.add_argument(
        '--use-existing-data',
        action='store_true',
        help='Use existing data files instead of collecting new data'
    )
    
    parser.add_argument(
        '--feature-type',
        type=str,
        default='tfidf',
        choices=['tfidf', 'embeddings', 'hybrid'],
        help='Type of features to extract'
    )
    
    parser.add_argument(
        '--use-smote',
        action='store_true',
        help='Use SMOTE for handling imbalanced data'
    )
    
    parser.add_argument(
        '--tune-hyperparameters',
        action='store_true',
        help='Perform hyperparameter tuning (time-consuming)'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = SentimentAnalysisPipeline()
    pipeline.run_complete_pipeline(
        use_existing_data=args.use_existing_data,
        feature_type=args.feature_type,
        use_smote=args.use_smote,
        tune_hyperparameters=args.tune_hyperparameters
    )


if __name__ == "__main__":
    main()