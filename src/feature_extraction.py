"""
Feature Extraction Module
Implements TF-IDF and embedding-based feature extraction.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import logging
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack, csr_matrix

import sys
sys.path.append('..')
from config.settings import FEATURE_CONFIG, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TFIDFFeatureExtractor:
    """Extract features using TF-IDF (Term Frequency-Inverse Document Frequency)."""
    
    def __init__(self, config: dict = None):
        """
        Initialize TF-IDF feature extractor.
        
        Args:
            config: TF-IDF configuration dictionary
        """
        self.config = config or FEATURE_CONFIG['tfidf']
        self.vectorizer = None
        self.feature_names = None
        
    def fit(self, texts: List[str]):
        """
        Fit TF-IDF vectorizer on training texts.
        
        Args:
            texts: List of preprocessed texts
        """
        logger.info("Fitting TF-IDF vectorizer...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.config['max_features'],
            ngram_range=self.config['ngram_range'],
            min_df=self.config['min_df'],
            max_df=self.config['max_df'],
            sublinear_tf=True,  # Apply sublinear tf scaling
            use_idf=True,
        )
        
        self.vectorizer.fit(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        logger.info(f"TF-IDF vectorizer fitted with {len(self.feature_names)} features")
        
    def transform(self, texts: List[str]) -> csr_matrix:
        """
        Transform texts to TF-IDF features.
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            Sparse matrix of TF-IDF features
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer must be fitted before transform")
        
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts: List[str]) -> csr_matrix:
        """
        Fit vectorizer and transform texts in one step.
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            Sparse matrix of TF-IDF features
        """
        self.fit(texts)
        return self.transform(texts)
    
    def get_top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        """
        Get top N features by average TF-IDF score.
        
        Args:
            n: Number of top features to return
            
        Returns:
            List of (feature, score) tuples
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer must be fitted first")
        
        # Get feature names and their IDF values
        idf_scores = self.vectorizer.idf_
        feature_scores = list(zip(self.feature_names, idf_scores))
        
        # Sort by IDF score (descending)
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        return feature_scores[:n]
    
    def save(self, filename: str = 'tfidf_vectorizer.pkl'):
        """Save fitted vectorizer to disk."""
        filepath = MODELS_DIR / filename
        joblib.dump(self.vectorizer, filepath)
        logger.info(f"TF-IDF vectorizer saved to {filepath}")
    
    def load(self, filename: str = 'tfidf_vectorizer.pkl'):
        """Load fitted vectorizer from disk."""
        filepath = MODELS_DIR / filename
        self.vectorizer = joblib.load(filepath)
        self.feature_names = self.vectorizer.get_feature_names_out()
        logger.info(f"TF-IDF vectorizer loaded from {filepath}")


class EmbeddingFeatureExtractor:
    """Extract features using sentence embeddings."""
    
    def __init__(self, config: dict = None):
        """
        Initialize embedding feature extractor.
        
        Args:
            config: Embedding configuration dictionary
        """
        self.config = config or FEATURE_CONFIG['embeddings']
        self.model = None
        self.embedding_dim = None
        
    def load_model(self):
        """Load pre-trained sentence transformer model."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.config['model_name']}")
            self.model = SentenceTransformer(self.config['model_name'])
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings (n_samples, embedding_dim)
        """
        self.load_model()
        
        logger.info(f"Encoding {len(texts)} texts to embeddings...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.config['batch_size'],
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        logger.info(f"Embeddings shape: {embeddings.shape}")
        return embeddings
    
    def compute_similarity(self, 
                          text1: str, 
                          text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        self.load_model()
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        emb1 = self.model.encode([text1])
        emb2 = self.model.encode([text2])
        
        return cosine_similarity(emb1, emb2)[0][0]


class HybridFeatureExtractor:
    """Combine TF-IDF and embedding features."""
    
    def __init__(self, 
                 tfidf_config: dict = None, 
                 embedding_config: dict = None):
        """
        Initialize hybrid feature extractor.
        
        Args:
            tfidf_config: TF-IDF configuration
            embedding_config: Embedding configuration
        """
        self.tfidf_extractor = TFIDFFeatureExtractor(tfidf_config)
        self.embedding_extractor = EmbeddingFeatureExtractor(embedding_config)
        
    def fit(self, texts: List[str]):
        """
        Fit TF-IDF vectorizer (embeddings don't need fitting).
        
        Args:
            texts: List of preprocessed texts
        """
        self.tfidf_extractor.fit(texts)
    
    def transform(self, 
                 texts: List[str], 
                 use_tfidf: bool = True,
                 use_embeddings: bool = True) -> np.ndarray:
        """
        Transform texts to combined features.
        
        Args:
            texts: List of texts
            use_tfidf: Whether to include TF-IDF features
            use_embeddings: Whether to include embedding features
            
        Returns:
            Combined feature matrix
        """
        features = []
        
        if use_tfidf:
            tfidf_features = self.tfidf_extractor.transform(texts)
            features.append(tfidf_features)
        
        if use_embeddings:
            embedding_features = self.embedding_extractor.encode(texts)
            features.append(embedding_features)
        
        # Combine features
        if len(features) == 1:
            return features[0]
        
        # Convert sparse to dense if needed and concatenate
        combined = hstack([
            f if hasattr(f, 'toarray') else csr_matrix(f) 
            for f in features
        ])
        
        return combined
    
    def fit_transform(self, 
                     texts: List[str],
                     use_tfidf: bool = True,
                     use_embeddings: bool = True) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            texts: List of texts
            use_tfidf: Whether to include TF-IDF features
            use_embeddings: Whether to include embedding features
            
        Returns:
            Combined feature matrix
        """
        if use_tfidf:
            self.fit(texts)
        
        return self.transform(texts, use_tfidf, use_embeddings)


class FeaturePipeline:
    """Complete feature extraction pipeline with statistical features."""
    
    def __init__(self):
        """Initialize feature pipeline."""
        self.tfidf_extractor = TFIDFFeatureExtractor()
        self.embedding_extractor = EmbeddingFeatureExtractor()
    
    def extract_statistical_features(self, 
                                    texts: List[str]) -> np.ndarray:
        """
        Extract statistical text features.
        
        Args:
            texts: List of texts
            
        Returns:
            Array of statistical features
        """
        features = []
        
        for text in texts:
            words = text.split()
            
            feat = {
                'text_length': len(text),
                'word_count': len(words),
                'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
                'unique_word_ratio': len(set(words)) / len(words) if words else 0,
            }
            
            features.append(list(feat.values()))
        
        return np.array(features)
    
    def extract_all_features(self, 
                           texts: List[str],
                           feature_types: List[str] = None) -> dict:
        """
        Extract all feature types.
        
        Args:
            texts: List of preprocessed texts
            feature_types: List of feature types to extract
                          ['tfidf', 'embeddings', 'statistical']
            
        Returns:
            Dictionary with feature matrices
        """
        if feature_types is None:
            feature_types = ['tfidf', 'embeddings', 'statistical']
        
        features = {}
        
        if 'tfidf' in feature_types:
            logger.info("Extracting TF-IDF features...")
            features['tfidf'] = self.tfidf_extractor.fit_transform(texts)
        
        if 'embeddings' in feature_types:
            logger.info("Extracting embedding features...")
            features['embeddings'] = self.embedding_extractor.encode(texts)
        
        if 'statistical' in feature_types:
            logger.info("Extracting statistical features...")
            features['statistical'] = self.extract_statistical_features(texts)
        
        return features


def main():
    """Example usage of feature extractors."""
    # Sample texts
    texts = [
        "machine learning artificial intelligence",
        "deep learning neural network",
        "natural language processing text analysis"
    ]
    
    print("=== TF-IDF Features ===")
    tfidf = TFIDFFeatureExtractor()
    tfidf_features = tfidf.fit_transform(texts)
    print(f"Shape: {tfidf_features.shape}")
    print(f"Top features: {tfidf.get_top_features(5)}")
    
    print("\n=== Embedding Features ===")
    embeddings = EmbeddingFeatureExtractor()
    emb_features = embeddings.encode(texts, show_progress=False)
    print(f"Shape: {emb_features.shape}")
    
    print("\n=== Similarity Test ===")
    sim = embeddings.compute_similarity(texts[0], texts[1])
    print(f"Similarity between text 1 and 2: {sim:.4f}")


if __name__ == "__main__":
    main()