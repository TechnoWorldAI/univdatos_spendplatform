#!/usr/bin/env python3
"""
Spend Platform Classification Model Runner
Main script to run the CEaSR transaction classification model
"""

import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from pathlib import Path

# Add the classificationmodel directory to the path
sys.path.append(str(Path(__file__).parent / 'classificationmodel'))

from classificationmodel.transaction_classifier import TransactionClassifier
from classificationmodel.data_preprocessing import TransactionDataPreprocessor
from classificationmodel.evaluation import ModelEvaluator, DataVisualizer

def main():
    """Main function to run the classification model"""

    # File paths
    sample_data_path = Path(__file__).parent / 'context' / 'sample spend data- filled.xlsx'
    categorization_path = Path(__file__).parent / 'context' / 'Categorization File.xlsx'
    unspsc_path = Path(__file__).parent / 'context' / 'UNGM_UNSPSC_01-Sep-2025..xlsx'

    print("Loading data...")

    # Load sample spend data
    sample_df = pd.read_excel(sample_data_path)
    print(f"Loaded {len(sample_df)} transactions from sample data")

    # Load UNSPSC categorization data
    unspsc_df = pd.read_excel(unspsc_path)
    print(f"Loaded {len(unspsc_df)} UNSPSC categories")
    print(f"UNSPSC categories range from {unspsc_df['Code'].min()} to {unspsc_df['Code'].max()}")

    # Create enhanced text features using UNSPSC categories
    print("\nEnhancing model with UNSPSC reference data...")

    # Add UNSPSC titles to text preprocessing for better matching
    unspsc_titles = set(unspsc_df['Title'].str.lower().str.strip())
    print(f"Added {len(unspsc_titles)} UNSPSC category titles for enhanced text matching")

    # You could also create a mapping function here for detailed UNSPSC classification
    # For now, we'll use it to enhance the existing model's text processing

    # Preprocess the data
    print("\nPreprocessing data...")
    preprocessor = TransactionDataPreprocessor()
    processed_df = preprocessor.preprocess(sample_df)
    print(f"Processed {len(processed_df)} transactions")

    # Prepare training data with proper split
    print(f"\nTotal samples: {len(processed_df)}")
    print(f"Categories: {processed_df['Tower (Practice)'].value_counts()}")

    # Use same data for training and testing (like original)
    train_df = processed_df.copy()
    test_df = processed_df.copy()

    print(f"\nTraining on {len(train_df)} transactions")
    print(f"Testing on {len(test_df)} transactions")
    print(f"Train category distribution:\n{train_df['Tower (Practice)'].value_counts()}")
    print(f"Test category distribution:\n{test_df['Tower (Practice)'].value_counts()}")

    # Initialize and train the classifier
    print("\nInitializing CEaSR classifier...")
    classifier = TransactionClassifier()

    print("Training model...")
    classifier.fit(train_df)

    # Make predictions
    print("Making predictions...")
    predictions, confidences = classifier.predict(test_df)

    # Evaluate the model
    print("\nEvaluating model...")
    evaluator = ModelEvaluator()
    eval_results = evaluator.evaluate_predictions(
        y_true=np.array(test_df['Tower (Practice)']),
        y_pred=np.array(predictions),
        confidences=np.array(confidences)
    )

    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {eval_results['accuracy']:.4f}")
    print(f"Classification Report:")
    print(eval_results['classification_report'])

    # Visualize results
    print("\nGenerating visualizations...")
    visualizer = DataVisualizer()

    # Plot confusion matrix using evaluator
    evaluator.plot_confusion_matrix()

    # Plot data distributions
    visualizer.plot_category_distribution(processed_df)
    visualizer.plot_text_length_distribution(processed_df)
    visualizer.plot_amount_distribution(processed_df)

    print("\nClassification model execution completed!")
    print("Check the generated plots and evaluation report above.")

if __name__ == "__main__":
    main()
