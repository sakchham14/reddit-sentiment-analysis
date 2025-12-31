"""
Visualization Module
Advanced plotting and visualization utilities.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from typing import List, Dict, Tuple
import logging

import sys
sys.path.append('..')
from config.settings import VIZ_CONFIG, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataVisualizer:
    """Comprehensive data visualization utilities."""
    
    def __init__(self):
        """Initialize visualizer with plotting settings."""
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette(VIZ_CONFIG['color_palette'])
        self.figsize = VIZ_CONFIG['figure_size']
        self.dpi = VIZ_CONFIG['dpi']
    
    def plot_sentiment_distribution(self, 
                                   sentiments: pd.Series,
                                   title: str = "Sentiment Distribution",
                                   save_path: str = None):
        """
        Plot distribution of sentiment classes.
        
        Args:
            sentiments: Series of sentiment labels
            title: Plot title
            save_path: Path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        sentiment_counts = sentiments.value_counts()
        colors = sns.color_palette('husl', len(sentiment_counts))
        
        ax1.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
        ax1.set_xlabel('Sentiment', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Sentiment Counts', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(sentiment_counts.values):
            ax1.text(i, v + max(sentiment_counts.values) * 0.01, 
                    str(v), ha='center', fontweight='bold')
        
        # Pie chart
        ax2.pie(sentiment_counts.values, labels=sentiment_counts.index, 
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Sentiment Proportions', fontsize=14, fontweight='bold')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_text_length_distribution(self,
                                     df: pd.DataFrame,
                                     text_column: str = 'full_text',
                                     sentiment_column: str = 'sentiment',
                                     save_path: str = None):
        """
        Plot text length distribution by sentiment.
        
        Args:
            df: DataFrame with text data
            text_column: Name of text column
            sentiment_column: Name of sentiment column
            save_path: Path to save the figure
        """
        # Calculate text lengths
        df['text_length'] = df[text_column].str.len()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Overall distribution
        ax1.hist(df['text_length'], bins=50, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Text Length (characters)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Overall Text Length Distribution', 
                     fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # By sentiment
        sentiments = df[sentiment_column].unique()
        for sentiment in sentiments:
            data = df[df[sentiment_column] == sentiment]['text_length']
            ax2.hist(data, bins=30, alpha=0.5, label=sentiment)
        
        ax2.set_xlabel('Text Length (characters)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Text Length by Sentiment', 
                     fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_wordcloud(self,
                      texts: List[str],
                      title: str = "Word Cloud",
                      max_words: int = 100,
                      save_path: str = None):
        """
        Generate and plot word cloud.
        
        Args:
            texts: List of text strings
            title: Plot title
            max_words: Maximum number of words to display
            save_path: Path to save the figure
        """
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=max_words,
            colormap='viridis',
            relative_scaling=0.5,
            min_font_size=10
        ).generate(combined_text)
        
        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_sentiment_wordclouds(self,
                                 df: pd.DataFrame,
                                 text_column: str = 'cleaned_text',
                                 sentiment_column: str = 'sentiment',
                                 save_path: str = None):
        """
        Generate word clouds for each sentiment class.
        
        Args:
            df: DataFrame with text and sentiment
            text_column: Name of text column
            sentiment_column: Name of sentiment column
            save_path: Path to save the figure
        """
        sentiments = df[sentiment_column].unique()
        n_sentiments = len(sentiments)
        
        fig, axes = plt.subplots(1, n_sentiments, figsize=(6*n_sentiments, 5))
        if n_sentiments == 1:
            axes = [axes]
        
        for idx, sentiment in enumerate(sentiments):
            texts = df[df[sentiment_column] == sentiment][text_column].tolist()
            combined_text = ' '.join(texts)
            
            wordcloud = WordCloud(
                width=600,
                height=400,
                background_color='white',
                max_words=50,
                colormap='viridis'
            ).generate(combined_text)
            
            axes[idx].imshow(wordcloud, interpolation='bilinear')
            axes[idx].axis('off')
            axes[idx].set_title(f'{sentiment.capitalize()} Sentiment', 
                              fontsize=14, fontweight='bold')
        
        plt.suptitle('Word Clouds by Sentiment', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_top_words(self,
                      texts: List[str],
                      top_n: int = 20,
                      title: str = "Top Words",
                      save_path: str = None):
        """
        Plot top N most frequent words.
        
        Args:
            texts: List of preprocessed texts
            top_n: Number of top words to display
            title: Plot title
            save_path: Path to save the figure
        """
        from collections import Counter
        
        # Count word frequencies
        all_words = ' '.join(texts).split()
        word_freq = Counter(all_words)
        top_words = word_freq.most_common(top_n)
        
        words, counts = zip(*top_words)
        
        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        y_pos = np.arange(len(words))
        ax.barh(y_pos, counts, color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        ax.set_xlabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_score_distribution(self,
                               df: pd.DataFrame,
                               score_column: str = 'score',
                               sentiment_column: str = 'sentiment',
                               save_path: str = None):
        """
        Plot Reddit score distribution by sentiment.
        
        Args:
            df: DataFrame with scores and sentiments
            score_column: Name of score column
            sentiment_column: Name of sentiment column
            save_path: Path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Box plot
        sentiments = sorted(df[sentiment_column].unique())
        data_to_plot = [df[df[sentiment_column] == s][score_column].values 
                       for s in sentiments]
        
        bp = ax1.boxplot(data_to_plot, labels=sentiments, patch_artist=True)
        
        # Color boxes
        colors = sns.color_palette('husl', len(sentiments))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_xlabel('Sentiment', fontsize=12)
        ax1.set_title('Score Distribution by Sentiment', 
                     fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Violin plot
        for i, sentiment in enumerate(sentiments):
            data = df[df[sentiment_column] == sentiment][score_column].values
            parts = ax2.violinplot([data], positions=[i], widths=0.7,
                                  showmeans=True, showmedians=True)
            
            # Color violin
            for pc in parts['bodies']:
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.7)
        
        ax2.set_xticks(range(len(sentiments)))
        ax2.set_xticklabels(sentiments)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_xlabel('Sentiment', fontsize=12)
        ax2.set_title('Score Distribution (Violin Plot)', 
                     fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_feature_importance(self,
                               feature_names: List[str],
                               importances: np.ndarray,
                               top_n: int = 20,
                               title: str = "Feature Importance",
                               save_path: str = None):
        """
        Plot feature importance for tree-based models.
        
        Args:
            feature_names: List of feature names
            importances: Array of importance scores
            top_n: Number of top features to display
            title: Plot title
            save_path: Path to save the figure
        """
        # Get top N features
        indices = np.argsort(importances)[-top_n:]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_importances, color='coral')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def create_eda_report(self,
                         df: pd.DataFrame,
                         text_column: str = 'full_text',
                         sentiment_column: str = 'sentiment',
                         output_dir: str = None):
        """
        Create complete exploratory data analysis report.
        
        Args:
            df: DataFrame with text data
            text_column: Name of text column
            sentiment_column: Name of sentiment column
            output_dir: Directory to save plots
        """
        if output_dir is None:
            output_dir = PROCESSED_DATA_DIR
        
        logger.info("Generating EDA report...")
        
        # 1. Sentiment distribution
        self.plot_sentiment_distribution(
            df[sentiment_column],
            save_path=output_dir / 'eda_sentiment_distribution.png'
        )
        plt.close()
        
        # 2. Text length distribution
        self.plot_text_length_distribution(
            df,
            text_column=text_column,
            sentiment_column=sentiment_column,
            save_path=output_dir / 'eda_text_length.png'
        )
        plt.close()
        
        # 3. Score distribution
        if 'score' in df.columns:
            self.plot_score_distribution(
                df,
                sentiment_column=sentiment_column,
                save_path=output_dir / 'eda_score_distribution.png'
            )
            plt.close()
        
        # 4. Word clouds by sentiment
        if 'cleaned_text' in df.columns:
            self.plot_sentiment_wordclouds(
                df,
                text_column='cleaned_text',
                sentiment_column=sentiment_column,
                save_path=output_dir / 'eda_wordclouds.png'
            )
            plt.close()
        
        logger.info(f"EDA report saved to {output_dir}")


def main():
    """Example usage of visualization functions."""
    # Create sample data
    sample_data = pd.DataFrame({
        'full_text': [
            'This is amazing! I love it!',
            'Terrible experience, very disappointed.',
            'It is okay, nothing special.',
            'Best purchase ever! Highly recommend!',
            'Worst product I have ever bought.'
        ] * 20,
        'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative'] * 20,
        'score': [10, -5, 0, 15, -10] * 20
    })
    
    # Initialize visualizer
    viz = DataVisualizer()
    
    # Plot sentiment distribution
    viz.plot_sentiment_distribution(sample_data['sentiment'])
    plt.show()
    
    # Create EDA report
    # viz.create_eda_report(sample_data)


if __name__ == "__main__":
    main()