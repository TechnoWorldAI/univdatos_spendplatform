"""
Evaluation and visualization utilities for the transaction classification system
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_curve, auc
)
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Comprehensive model evaluation utilities"""

    def __init__(self):
        self.results = {}

    def evaluate_predictions(self, y_true: np.ndarray,
                           y_pred: np.ndarray,
                           confidences: np.ndarray,
                           class_names: Optional[List[str]] = None) -> Dict:
        """Comprehensive evaluation of predictions"""

        results = {}

        # Basic metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)

        # Classification report
        results['classification_report'] = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        )

        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred)

        # Confidence analysis
        results['confidence_analysis'] = self._analyze_confidence(
            y_true, y_pred, confidences
        )

        # Per-class metrics
        precision, recall, fscore, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )

        results['per_class_metrics'] = {
            'precision': precision,
            'recall': recall,
            'fscore': fscore,
            'support': support
        }

        self.results = results
        return results

    def _analyze_confidence(self, y_true: np.ndarray,
                          y_pred: np.ndarray,
                          confidences: np.ndarray) -> Dict:
        """Analyze prediction confidence"""

        analysis = {}

        # Overall confidence statistics
        analysis['mean_confidence'] = np.mean(confidences)
        analysis['std_confidence'] = np.std(confidences)
        analysis['min_confidence'] = np.min(confidences)
        analysis['max_confidence'] = np.max(confidences)

        # Confidence vs accuracy analysis
        confidence_bins = np.linspace(0, 1, 11)
        bin_accuracies = []

        for i in range(len(confidence_bins) - 1):
            mask = (confidences >= confidence_bins[i]) & (confidences < confidence_bins[i + 1])
            if mask.sum() > 0:
                bin_accuracy = accuracy_score(y_true[mask], y_pred[mask])
            else:
                bin_accuracy = 0.0
            bin_accuracies.append(bin_accuracy)

        analysis['confidence_bins'] = confidence_bins[:-1]
        analysis['bin_accuracies'] = bin_accuracies

        # High confidence performance
        high_conf_thresholds = [0.7, 0.8, 0.9]
        for threshold in high_conf_thresholds:
            high_conf_mask = confidences >= threshold
            if high_conf_mask.sum() > 0:
                analysis[f'high_conf_accuracy_{int(threshold*100)}'] = accuracy_score(
                    y_true[high_conf_mask], y_pred[high_conf_mask]
                )
                analysis[f'high_conf_coverage_{int(threshold*100)}'] = high_conf_mask.mean()
            else:
                analysis[f'high_conf_accuracy_{int(threshold*100)}'] = 0.0
                analysis[f'high_conf_coverage_{int(threshold*100)}'] = 0.0

        return analysis

    def plot_confidence_analysis(self, save_path: Optional[str] = None):
        """Plot confidence vs accuracy analysis"""
        if 'confidence_analysis' not in self.results:
            print("No confidence analysis available")
            return

        conf_analysis = self.results['confidence_analysis']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Confidence distribution
        ax1.hist(conf_analysis['confidence_bins'], bins=10, alpha=0.7, edgecolor='black')
        ax1.set_title('Prediction Confidence Distribution')
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Frequency')

        # Confidence vs Accuracy
        ax2.plot(conf_analysis['confidence_bins'], conf_analysis['bin_accuracies'],
                marker='o', linestyle='-', linewidth=2, markersize=8)
        ax2.set_title('Confidence vs Accuracy')
        ax2.set_xlabel('Confidence Bin')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confidence analysis plot saved to {save_path}")

        plt.show()

    def plot_confusion_matrix(self, class_names: Optional[List[str]] = None,
                            save_path: Optional[str] = None):
        """Plot confusion matrix"""
        if 'confusion_matrix' not in self.results:
            print("No confusion matrix available")
            return

        cm = self.results['confusion_matrix']

        plt.figure(figsize=(10, 8))
        if class_names:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix plot saved to {save_path}")

        plt.show()

    def plot_class_performance(self, save_path: Optional[str] = None):
        """Plot per-class performance metrics"""
        if 'per_class_metrics' not in self.results:
            print("No per-class metrics available")
            return

        metrics = self.results['per_class_metrics']

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(metrics['precision']))
        width = 0.25

        ax.bar(x - width, metrics['precision'], width, label='Precision', alpha=0.8)
        ax.bar(x, metrics['recall'], width, label='Recall', alpha=0.8)
        ax.bar(x + width, metrics['fscore'], width, label='F1-Score', alpha=0.8)

        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class performance plot saved to {save_path}")

        plt.show()

    def generate_report(self, filepath: str):
        """Generate comprehensive evaluation report"""
        report = f"""
# Transaction Classification Model Evaluation Report

## Overall Performance
- **Accuracy**: {self.results.get('accuracy', 'N/A'):.4f}

## Confidence Analysis
- **Mean Confidence**: {self.results.get('confidence_analysis', {}).get('mean_confidence', 'N/A'):.4f}
- **High Confidence Accuracy (70%)**: {self.results.get('confidence_analysis', {}).get('high_conf_accuracy_70', 'N/A'):.4f}
- **High Confidence Coverage (70%)**: {self.results.get('confidence_analysis', {}).get('high_conf_coverage_70', 'N/A'):.4f}

## Classification Report
{self._format_classification_report()}
"""

        with open(filepath, 'w') as f:
            f.write(report)

        print(f"Evaluation report saved to {filepath}")

    def _format_classification_report(self) -> str:
        """Format classification report for the report"""
        if 'classification_report' not in self.results:
            return "No classification report available"

        report = self.results['classification_report']
        formatted = ""

        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                formatted += f"\n### {class_name}\n"
                formatted += f"- **Precision**: {metrics.get('precision', 'N/A'):.4f}\n"
                formatted += f"- **Recall**: {metrics.get('recall', 'N/A'):.4f}\n"
                formatted += f"- **F1-Score**: {metrics.get('f1-score', 'N/A'):.4f}\n"
                formatted += f"- **Support**: {metrics.get('support', 'N/A')}\n"

        return formatted


class DataVisualizer:
    """Data visualization utilities"""

    def __init__(self):
        self.style = 'seaborn-v0_8'
        plt.style.use(self.style)

    def plot_category_distribution(self, df: pd.DataFrame,
                                 category_column: str = 'Tower (Practice)',
                                 save_path: Optional[str] = None):
        """Plot category distribution"""
        plt.figure(figsize=(12, 6))

        category_counts = df[category_column].value_counts()

        sns.barplot(x=category_counts.index, y=category_counts.values)
        plt.title(f'Distribution of {category_column}')
        plt.xlabel(category_column)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Category distribution plot saved to {save_path}")

        plt.show()

    def plot_text_length_distribution(self, df: pd.DataFrame,
                                text_columns: Optional[List[str]] = None,
                                    save_path: Optional[str] = None):
        """Plot text length distributions"""
        if text_columns is None:
            text_columns = ['INV_ITEM_DESC', 'SUPPLIER_NAME']

        fig, axes = plt.subplots(1, len(text_columns), figsize=(15, 5))

        if len(text_columns) == 1:
            axes = [axes]

        for i, col in enumerate(text_columns):
            if col in df.columns:
                lengths = df[col].astype(str).str.len()
                sns.histplot(x=lengths, ax=axes[i], bins=30)
                axes[i].set_title(f'{col} Length Distribution')
                axes[i].set_xlabel('Length')
                axes[i].set_ylabel('Frequency')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Text length distribution plot saved to {save_path}")

        plt.show()

    def plot_amount_distribution(self, df: pd.DataFrame,
                               amount_column: str = 'Item Invoice Value',
                               save_path: Optional[str] = None):
        """Plot amount distribution"""
        plt.figure(figsize=(12, 6))

        if amount_column in df.columns:
            amounts = df[amount_column].abs()

            # Log scale for better visualization
            plt.subplot(1, 2, 1)
            sns.histplot(x=amounts, bins=50)
            plt.title(f'{amount_column} Distribution')
            plt.xlabel('Amount')
            plt.ylabel('Frequency')

            plt.subplot(1, 2, 2)
            sns.histplot(np.log1p(amounts), bins=50)
            plt.title(f'{amount_column} Distribution (Log Scale)')
            plt.xlabel('Log Amount')
            plt.ylabel('Frequency')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Amount distribution plot saved to {save_path}")

        plt.show()

    def plot_feature_correlations(self, df: pd.DataFrame,
                                numeric_columns: Optional[List[str]] = None,
                                save_path: Optional[str] = None):
        """Plot feature correlations"""
        if numeric_columns is None:
            numeric_columns = ['Item Invoice Value', 'PO Price', 'Unit Price',
                             'Item Qty Invoiced', 'supplier_length', 'desc_length']

        existing_cols = [col for col in numeric_columns if col in df.columns]

        if len(existing_cols) > 1:
            corr_matrix = df[existing_cols].corr()

            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Feature correlation plot saved to {save_path}")

            plt.show()
        else:
            print("Not enough numeric columns for correlation analysis")


def create_model_comparison_report(models_results: Dict[str, Dict],
                                 filepath: str):
    """Create a comparison report for multiple models"""
    report = "# Model Comparison Report\n\n"

    for model_name, results in models_results.items():
        report += f"## {model_name}\n"
        report += f"- **Accuracy**: {results.get('accuracy', 'N/A'):.4f}\n"
        if 'confidence_analysis' in results:
            conf = results['confidence_analysis']
            report += f"- **High Conf Accuracy (70%)**: {conf.get('high_conf_accuracy_70', 'N/A'):.4f}\n"
            report += f"- **High Conf Coverage (70%)**: {conf.get('high_conf_coverage_70', 'N/A'):.4f}\n"
        report += "\n"

    with open(filepath, 'w') as f:
        f.write(report)

    print(f"Model comparison report saved to {filepath}")
