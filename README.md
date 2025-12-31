# Reddit Sentiment Analysis Project

A comprehensive machine learning project for sentiment analysis of Reddit posts and comments using multiple feature extraction techniques and ML models.

## 🎯 Project Overview

This project implements an end-to-end sentiment analysis pipeline that:
- Collects real-time data from Reddit using PRAW API
- Performs comprehensive text preprocessing
- Compares TF-IDF and embedding-based feature extraction
- Trains and evaluates multiple ML models
- Provides detailed performance metrics and visualizations

## 🏗️ Project Structure

```
reddit-sentiment-analysis/
├── config/
│   ├── __init__.py
│   └── settings.py              # Configuration settings
├── data/
│   ├── raw/                     # Raw Reddit data
│   ├── processed/               # Cleaned and preprocessed data
│   └── models/                  # Saved trained models
├── src/
│   ├── __init__.py
│   ├── data_collection.py       # Reddit API data scraping
│   ├── preprocessing.py         # Text preprocessing pipeline
│   ├── feature_extraction.py    # TF-IDF and embeddings
│   ├── models.py                # ML model implementations
│   ├── evaluation.py            # Model evaluation utilities
│   └── visualization.py         # Plotting and visualization
├── notebooks/
│   └── exploratory_analysis.ipynb
├── tests/
│   └── test_preprocessing.py
├── requirements.txt
├── setup.py
├── README.md
└── main.py                      # Main execution script
```

## 🚀 Features

### Data Collection
- **PRAW Integration**: Seamless Reddit API integration
- **Multi-subreddit Support**: Collect from multiple subreddits
- **Flexible Filtering**: Time-based and score-based filtering
- **Batch Processing**: Efficient collection of posts and comments

### Text Preprocessing
- **Comprehensive Cleaning**: URL, email, number removal
- **Tokenization**: Word-level tokenization with NLTK
- **Stopword Removal**: Customizable stopword lists
- **Lemmatization**: POS-tagged lemmatization for accuracy
- **Contraction Expansion**: Handle common English contractions

### Feature Extraction
- **TF-IDF**: Term Frequency-Inverse Document Frequency with n-grams
- **Sentence Embeddings**: State-of-the-art transformer-based embeddings
- **Hybrid Features**: Combine TF-IDF and embeddings
- **Statistical Features**: Text length, word count, and more

### Machine Learning Models
- **Logistic Regression**: Fast baseline model
- **Random Forest**: Ensemble learning approach
- **Support Vector Machine**: Kernel-based classification
- **Naive Bayes**: Probabilistic classifier
- **Gradient Boosting**: Advanced ensemble method

### Evaluation & Analysis
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score
- **Per-class Analysis**: Detailed metrics for each sentiment class
- **Confusion Matrix**: Visual representation of predictions
- **ROC Curves**: Binary classification performance
- **Model Comparison**: Side-by-side model comparison

## 📋 Prerequisites

- Python 3.8+
- Reddit API credentials (client_id, client_secret)
- 8GB RAM minimum (16GB recommended for large datasets)

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/sakchham14/reddit-sentiment-analysis.git
cd reddit-sentiment-analysis
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

5. **Set up Reddit API credentials**

Create a `.env` file in the project root:
```env
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=sentiment_analysis_bot/1.0
REDDIT_USERNAME=your_username
REDDIT_PASSWORD=your_password
```

To get Reddit API credentials:
1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Select "script" as the app type
4. Fill in the required fields
5. Copy the client_id and client_secret

## 🎮 Usage

### Quick Start

Run the complete pipeline with default settings:
```bash
python main.py
```

### Advanced Usage

**Use existing data (skip collection):**
```bash
python main.py --use-existing-data
```

**Use embedding features instead of TF-IDF:**
```bash
python main.py --feature-type embeddings
```

**Handle imbalanced data with SMOTE:**
```bash
python main.py --use-smote
```

**Perform hyperparameter tuning:**
```bash
python main.py --tune-hyperparameters
```

**Combine multiple options:**
```bash
python main.py --use-existing-data --feature-type hybrid --use-smote
```

### Individual Module Usage

**Data Collection:**
```python
from src.data_collection import RedditDataCollector

collector = RedditDataCollector()
posts_df, comments_df = collector.collect_dataset(
    subreddits=['technology', 'news'],
    post_limit=100
)
```

**Text Preprocessing:**
```python
from src.preprocessing import TextPreprocessor

preprocessor = TextPreprocessor()
cleaned_text = preprocessor.preprocess("Your text here!")
```

**Feature Extraction:**
```python
from src.feature_extraction import TFIDFFeatureExtractor

extractor = TFIDFFeatureExtractor()
features = extractor.fit_transform(texts)
```

**Model Training:**
```python
from src.models import LogisticRegressionModel

model = LogisticRegressionModel()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## 📊 Results

The project generates several outputs in the `data/processed/` directory:

- `processed_data.csv`: Cleaned and preprocessed text data
- `evaluation_results.csv`: Detailed metrics for all models
- `model_comparison.png`: Visual comparison of model performance
- `confusion_matrix_*.png`: Confusion matrices for each model

### Sample Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.85 | 0.84 | 0.85 | 0.84 |
| Random Forest | 0.83 | 0.82 | 0.83 | 0.82 |
| SVM | 0.86 | 0.85 | 0.86 | 0.85 |
| Naive Bayes | 0.79 | 0.78 | 0.79 | 0.78 |

*Note: Actual results will vary based on your data*

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src tests/
```

## 📈 Model Comparison

### TF-IDF vs Embeddings

**TF-IDF Advantages:**
- Faster training and inference
- Interpretable features (word importance)
- Works well with linear models
- Lower memory footprint

**Embeddings Advantages:**
- Captures semantic similarity
- Better generalization
- Context-aware representations
- State-of-the-art performance

### Recommended Approach

For production use, we recommend:
1. Start with TF-IDF for baseline
2. Experiment with embeddings for improved performance
3. Use hybrid features for best of both worlds
4. Consider computational constraints

## 🔧 Configuration

Modify `config/settings.py` to customize:

- **Data Collection**: Subreddits, limits, time filters
- **Preprocessing**: Stopwords, lemmatization, cleaning rules
- **Features**: TF-IDF parameters, embedding models
- **Models**: Hyperparameters, cross-validation settings

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PRAW library for Reddit API access
- scikit-learn for machine learning tools
- sentence-transformers for embeddings
- NLTK for natural language processing

## 📧 Contact

Your Name - your.email@example.com

Project Link: https://github.com/yourusername/reddit-sentiment-analysis

## 🎓 Academic Use

This project is suitable for:
- Final year CS projects
- Machine learning coursework
- NLP research projects
- Portfolio demonstrations

### Citation

If you use this project in your research, please cite:
```
@software{reddit_sentiment_analysis,
  author = {Sakchham},
  title = {Reddit Sentiment Analysis: A Comprehensive ML Pipeline},
  year = {2024},
  url = {https://github.com/sakchham14/reddit-sentiment-analysis}
}
```

## 📚 Further Reading

- [PRAW Documentation](https://praw.readthedocs.io/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Sentiment Analysis Tutorial](https://realpython.com/sentiment-analysis-python/)
- [TF-IDF Explained](https://monkeylearn.com/blog/what-is-tf-idf/)

## 🐛 Known Issues

- Large datasets may require significant RAM for embedding extraction
- Reddit API rate limiting may slow down data collection
- Some models require dense matrices (convert sparse with `.toarray()`)

## 🗺️ Roadmap

- [ ] Add deep learning models (LSTM, BERT)
- [ ] Implement real-time sentiment monitoring
- [ ] Add multilingual support
- [ ] Create web interface with Flask/FastAPI
- [ ] Add sentiment aspect analysis
- [ ] Implement active learning for labeling

---

**⭐ If you find this project helpful, please consider giving it a star!**