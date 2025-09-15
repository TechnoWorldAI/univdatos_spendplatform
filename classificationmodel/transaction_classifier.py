"""
Large Scale Personalized Classification of Financial Transactions
Based on the CEaSR (Confidence-based Ensemble of Association Strength Rankers) approach
from the paper by Lesner, Ran, Rukonic, and Wang (2020)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict
import pickle
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import re
from collections import defaultdict, Counter
import pickle
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add NLTK for better text processing
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available, using basic text processing")

class TransactionClassifier:
    """
    CEaSR-based transaction classification system
    Implements Confidence-based Ensemble of Association Strength Rankers
    """

    def __init__(self, config: Optional[Dict] = None, unspsc_data: Optional[pd.DataFrame] = None):
        self.config = config or {
            'min_samples': 5,
            'confidence_threshold': 0.3,  # Lower threshold for better coverage
            'max_features': 1000,
            'random_state': 42,
            'use_cross_validation': True,
            'cv_folds': 5
        }

        # Initialize components
        self.label_encoders = {}
        self.association_models = {}
        self.confidence_models = {}
        self.ensemble_weights = {}
        self.vectorizers = {}

        # Data structures for association strength
        self.supplier_category_counts = defaultdict(Counter)
        self.description_category_counts = defaultdict(Counter)
        self.tower_category_counts = defaultdict(Counter)

        # UNSPSC enhancement
        self.unspsc_data = unspsc_data
        self.unspsc_mapping = {}
        self.unspsc_text_features = set()
        if unspsc_data is not None:
            self._prepare_unspsc_data()

    def _prepare_unspsc_data(self):
        """Prepare UNSPSC data for enhanced text matching and classification"""
        if self.unspsc_data is None:
            return

        print("Preparing UNSPSC data for enhanced classification...")

        # Create mapping from titles to codes
        self.unspsc_mapping = dict(zip(
            self.unspsc_data['Title'].str.lower().str.strip(),
            self.unspsc_data['Code']
        ))

        # Extract key terms from UNSPSC titles for enhanced text matching
        for title in self.unspsc_data['Title'].dropna():
            # Tokenize and clean
            if NLTK_AVAILABLE:
                try:
                    tokens = word_tokenize(title.lower())
                    tokens = [t for t in tokens if t not in stopwords.words('english') and len(t) > 2]
                    self.unspsc_text_features.update(tokens)
                except:
                    # Fallback: simple split
                    tokens = [t.strip() for t in title.lower().split() if len(t.strip()) > 2]
                    self.unspsc_text_features.update(tokens)
            else:
                tokens = [t.strip() for t in title.lower().split() if len(t.strip()) > 2]
                self.unspsc_text_features.update(tokens)

        print(f"Prepared {len(self.unspsc_mapping)} UNSPSC mappings")
        print(f"Extracted {len(self.unspsc_text_features)} UNSPSC text features")

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the transaction data"""
        print("Preprocessing data...")

        # Clean text fields
        text_columns = ['INV_ITEM_DESC', 'SUPPLIER_NAME', 'Material Item Name', 'Order description']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str).str.lower()
                df[col] = df[col].apply(self._clean_text)

        # Create target variable from Tower (Practice) as it's the most complete category
        df['target_category'] = df['Tower (Practice)'].fillna('Unknown')

        # Filter out unknown categories
        df = df[df['target_category'] != 'Unknown'].copy()

        print(f"Preprocessed data shape: {df.shape}")
        print(f"Unique categories: {df['target_category'].nunique()}")

        return df

    def _clean_text(self, text: str) -> str:
        """Clean text data with advanced preprocessing"""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers, keep spaces and letters
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        # Tokenize and process with NLTK if available
        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text)
                # Remove stop words
                stop_words = set(stopwords.words('english'))
                tokens = [token for token in tokens if token not in stop_words and len(token) > 2]

                # Lemmatize
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(token) for token in tokens]

                text = ' '.join(tokens)
            except:
                # Fallback to basic processing
                tokens = [t for t in text.split() if len(t) > 2]
                text = ' '.join(tokens)
        else:
            tokens = [t for t in text.split() if len(t) > 2]
            text = ' '.join(tokens)

        return text.strip()

    def build_association_strength_models(self, df: pd.DataFrame):
        """Build association strength models based on historical data"""
        print("Building association strength models...")

        # Build supplier-category associations
        supplier_category = df.groupby(['SUPPLIER_NAME', 'target_category']).size().reset_index(name='count')
        for idx, row in supplier_category.iterrows():
            supplier = row['SUPPLIER_NAME']
            category = row['target_category']
            count = row['count']
            if supplier and supplier != '':
                self.supplier_category_counts[supplier][category] = count

        # Build description-category associations
        desc_category = df.groupby(['INV_ITEM_DESC', 'target_category']).size().reset_index(name='count')
        for idx, row in desc_category.iterrows():
            desc = row['INV_ITEM_DESC']
            category = row['target_category']
            count = row['count']
            if desc and desc != '':
                self.description_category_counts[desc][category] = count

        # Build tower-category associations (for cross-validation)
        tower_category = df.groupby(['Tower (Practice)', 'target_category']).size().reset_index(name='count')
        for idx, row in tower_category.iterrows():
            tower = row['Tower (Practice)']
            category = row['target_category']
            count = row['count']
            if tower and tower != '':
                self.tower_category_counts[tower][category] = count

        print(f"Built associations for {len(self.supplier_category_counts)} suppliers")
        print(f"Built associations for {len(self.description_category_counts)} descriptions")

    def _calculate_association_strength(self, text: str, category: str, text_type: str) -> float:
        """Calculate association strength with improved smoothing"""
        if not text or text == '':
            return 0.0

        if text_type == 'supplier':
            counts = self.supplier_category_counts.get(text, {})
        elif text_type == 'description':
            counts = self.description_category_counts.get(text, {})
        else:
            return 0.0

        # Get counts
        category_count = counts.get(category, 0)
        total_count = sum(counts.values())

        if total_count == 0:
            return 1.0 / len(self.label_encoders['target'].classes_)  # Return uniform probability

        # Simple Laplace smoothing
        smoothed_prob = (category_count + 1) / (total_count + len(self.label_encoders['target'].classes_))

        return smoothed_prob

    def create_weak_predictors(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create multiple weak ranking predictors"""
        print("Creating weak predictors...")

        predictors = {}

        # Predictor 1: Supplier-based association
        predictors['supplier'] = self._create_supplier_predictor(df)

        # Predictor 2: Description-based association
        predictors['description'] = self._create_description_predictor(df)

        # Predictor 3: Combined text features
        predictors['text_combined'] = self._create_text_predictor(df)

        # Predictor 4: UNSPSC-based features (if available)
        unspsc_pred = self._create_unspsc_predictor(df)
        if not unspsc_pred.empty:
            predictors['unspsc'] = unspsc_pred

        # Predictor 5: Length-based features
        predictors['length_features'] = self._create_length_predictor(df)

        print(f"Created {len(predictors)} weak predictors")
        return predictors

    def _create_supplier_predictor(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create supplier-based predictor"""
        # Use all categories from training data (label encoder) to ensure consistency
        if hasattr(self, 'label_encoders') and 'target' in self.label_encoders:
            categories = list(self.label_encoders['target'].classes_)
        else:
            categories = sorted(df['target_category'].unique())

        features = []

        for idx, row in df.iterrows():
            feature_row = {}
            supplier = row['SUPPLIER_NAME']
            if supplier in self.supplier_category_counts:
                supplier_counts = self.supplier_category_counts[supplier]
                total = sum(supplier_counts.values())

                # Calculate association strengths
                for category in categories:
                    count = supplier_counts.get(category, 0)
                    # Add Laplace smoothing
                    strength = (count + 1) / (total + len(categories))
                    feature_row[f'supplier_{category}'] = strength
            else:
                # Default values for unknown suppliers
                for category in categories:
                    feature_row[f'supplier_{category}'] = 1.0 / len(categories)

            features.append(feature_row)

        return pd.DataFrame(features)

    def _create_description_predictor(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create description-based predictor"""
        # Use all categories from training data (label encoder) to ensure consistency
        if hasattr(self, 'label_encoders') and 'target' in self.label_encoders:
            categories = list(self.label_encoders['target'].classes_)
        else:
            categories = sorted(df['target_category'].unique())

        features = []

        for idx, row in df.iterrows():
            feature_row = {}
            desc = row['INV_ITEM_DESC']
            if desc in self.description_category_counts:
                desc_counts = self.description_category_counts[desc]
                total = sum(desc_counts.values())

                for category in categories:
                    count = desc_counts.get(category, 0)
                    strength = (count + 1) / (total + len(categories))
                    feature_row[f'desc_{category}'] = strength
            else:
                for category in categories:
                    feature_row[f'desc_{category}'] = 1.0 / len(categories)

            features.append(feature_row)

        return pd.DataFrame(features)

    def _create_text_predictor(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create combined text-based predictor using TF-IDF"""
        # Combine text features
        df['combined_text'] = df['INV_ITEM_DESC'] + ' ' + df['SUPPLIER_NAME'] + ' ' + df['Material Item Name'].fillna('')

        # Create TF-IDF features with improved parameters
        if not hasattr(self, 'vectorizers') or 'combined' not in self.vectorizers:
            self.vectorizers['combined'] = TfidfVectorizer(
                max_features=1000,  # Increased features
                stop_words='english',
                ngram_range=(1, 2),  # Include bigrams
                min_df=2,  # Ignore terms that appear in less than 2 documents
                max_df=0.9,  # Ignore terms that appear in more than 90% of documents
                sublinear_tf=True  # Apply sublinear tf scaling
            )
            X_text = self.vectorizers['combined'].fit_transform(df['combined_text'])
        else:
            X_text = self.vectorizers['combined'].transform(df['combined_text'])

        # Convert to DataFrame
        feature_names = [f'text_{i}' for i in range(X_text.shape[1])]
        text_features = pd.DataFrame(X_text.toarray(), columns=feature_names)

        return text_features

    def _create_unspsc_predictor(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create UNSPSC-based predictor for detailed sub-classification"""
        if self.unspsc_data is None:
            # Return empty dataframe if no UNSPSC data
            return pd.DataFrame()

        features = []

        for idx, row in df.iterrows():
            feature_row = {}

            # Combine all text fields for UNSPSC matching
            combined_text = ' '.join([
                str(row.get('INV_ITEM_DESC', '')),
                str(row.get('SUPPLIER_NAME', '')),
                str(row.get('Material Item Name', '')),
                str(row.get('Order description', ''))
            ]).lower()

            # Find best UNSPSC matches using improved scoring
            best_matches = []
            for title, code in self.unspsc_mapping.items():
                # Improved text overlap scoring with TF-IDF like weighting
                title_words = set(title.lower().split())
                text_words = set(combined_text.split())

                if not title_words:
                    continue

                # Calculate Jaccard similarity
                intersection = title_words.intersection(text_words)
                union = title_words.union(text_words)
                jaccard_score = len(intersection) / len(union) if union else 0

                # Calculate overlap ratio
                overlap_score = len(intersection) / len(title_words)

                # Combine scores with weighting
                combined_score = 0.7 * overlap_score + 0.3 * jaccard_score

                if combined_score > 0.1:  # Minimum threshold
                    best_matches.append((code, combined_score))

            # Sort by score and take top matches
            best_matches.sort(key=lambda x: x[1], reverse=True)
            top_matches = best_matches[:5]  # Top 5 matches

            # Create features for top matches
            for i, (code, score) in enumerate(top_matches):
                feature_row[f'unspsc_code_{i}'] = code
                feature_row[f'unspsc_score_{i}'] = score

            # Fill missing features with a default UNSPSC code
            for i in range(5):
                if f'unspsc_code_{i}' not in feature_row:
                    feature_row[f'unspsc_code_{i}'] = '00000000'  # Default UNSPSC code for uncategorized
                    feature_row[f'unspsc_score_{i}'] = 0.0

            features.append(feature_row)

        unspsc_df = pd.DataFrame(features)

        # Convert categorical UNSPSC codes to numerical features using label encoding
        for i in range(5):
            code_col = f'unspsc_code_{i}'
            if code_col in unspsc_df.columns:
                # Use label encoding for UNSPSC codes
                unique_codes = unspsc_df[code_col].unique()
                code_mapping = {code: idx for idx, code in enumerate(unique_codes)}
                unspsc_df[f'{code_col}_num'] = unspsc_df[code_col].map(code_mapping)

        return unspsc_df

    def _create_association_predictor(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create association strength-based predictor"""
        # Use all categories from training data (label encoder) to ensure consistency
        if hasattr(self, 'label_encoders') and 'target' in self.label_encoders:
            categories = list(self.label_encoders['target'].classes_)
        else:
            categories = sorted(df['target_category'].unique())

        features = []

        for idx, row in df.iterrows():
            feature_row = {}

            # Supplier association features
            supplier = str(row.get('SUPPLIER_NAME', ''))
            if supplier in self.supplier_category_counts:
                supplier_counts = self.supplier_category_counts[supplier]
                total_supplier = sum(supplier_counts.values())
                for category in categories:
                    feature_row[f'supplier_{category}'] = supplier_counts.get(category, 0) / total_supplier if total_supplier > 0 else 0

            # Description association features
            desc = str(row.get('INV_ITEM_DESC', ''))
            if desc in self.description_category_counts:
                desc_counts = self.description_category_counts[desc]
                total_desc = sum(desc_counts.values())
                for category in categories:
                    feature_row[f'desc_{category}'] = desc_counts.get(category, 0) / total_desc if total_desc > 0 else 0

            # Tower association features
            tower = str(row.get('Tower (Practice)', ''))
            if tower in self.tower_category_counts:
                tower_counts = self.tower_category_counts[tower]
                total_tower = sum(tower_counts.values())
                for category in categories:
                    feature_row[f'tower_{category}'] = tower_counts.get(category, 0) / total_tower if total_tower > 0 else 0

            features.append(feature_row)

        return pd.DataFrame(features).fillna(0)

    def _create_length_predictor(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create length-based features predictor"""
        features = []

        for idx, row in df.iterrows():
            feature_row = {
                'desc_length': len(str(row['INV_ITEM_DESC'])),
                'supplier_length': len(str(row['SUPPLIER_NAME'])),
                'total_text_length': len(str(row['INV_ITEM_DESC'])) + len(str(row['SUPPLIER_NAME'])),
                'word_count_desc': len(str(row['INV_ITEM_DESC']).split()),
                'word_count_supplier': len(str(row['SUPPLIER_NAME']).split())
            }
            features.append(feature_row)

        return pd.DataFrame(features)

    def train_confidence_models(self, predictors: Dict[str, pd.DataFrame], y_true: pd.Series):
        """Train confidence models for each predictor"""
        print("Training confidence models...")

        for predictor_name, X_pred in predictors.items():
            print(f"Training confidence model for {predictor_name}...")

            # Split data for training confidence model
            X_train, X_test, y_train, y_test = train_test_split(
                X_pred, y_true, test_size=0.2, random_state=self.config['random_state']
            )

            # Train base model with improved parameters
            if predictor_name == 'text_combined':
                base_model = RandomForestClassifier(
                    n_estimators=100,  # Moderate number of trees
                    max_depth=6,  # Moderate depth
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=self.config['random_state'],
                    n_jobs=-1
                )
            elif predictor_name == 'unspsc':
                base_model = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=4,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=self.config['random_state'],
                    n_jobs=-1
                )
            else:
                base_model = LogisticRegression(
                    random_state=self.config['random_state'],
                    C=0.1,  # Stronger regularization
                    max_iter=1000,
                    class_weight='balanced'  # Handle class imbalance
                )

            base_model.fit(X_train, y_train)

            # Get predictions and probabilities
            pred_probs = base_model.predict_proba(X_test)
            predictions = base_model.predict(X_test)

            # Create confidence training data
            confidence_features = []
            confidence_labels = []

            for i, (pred, true_label) in enumerate(zip(predictions, y_test)):
                # Use top-k predictions as features for confidence model
                top_k = 3
                sorted_indices = np.argsort(pred_probs[i])[::-1][:top_k]
                top_probs = pred_probs[i][sorted_indices]

                feature_row = list(top_probs)
                confidence_features.append(feature_row)

                # Confidence label: 1 if prediction is correct, 0 otherwise
                confidence_labels.append(1 if pred == true_label else 0)

            # Train confidence model with improved parameters
            conf_model = LogisticRegression(
                random_state=self.config['random_state'],
                C=0.5,  # Stronger regularization for confidence model
                max_iter=1000
            )
            conf_model.fit(confidence_features, confidence_labels)

            # Store models
            self.association_models[predictor_name] = base_model
            self.confidence_models[predictor_name] = conf_model

    def ensemble_predict(self, predictors: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """Make ensemble predictions using simple majority voting"""
        print("Making ensemble predictions...")

        all_predictions = []

        for predictor_name, X_pred in predictors.items():
            if predictor_name in self.association_models:
                base_model = self.association_models[predictor_name]
                predictions = base_model.predict(X_pred)
                all_predictions.append(predictions)

        # Simple majority voting
        all_predictions = np.array(all_predictions)
        final_predictions = []

        for i in range(len(all_predictions[0])):
            predictor_votes = all_predictions[:, i]
            # Use Counter to find most common prediction
            from collections import Counter
            most_common = Counter(predictor_votes).most_common(1)[0][0]
            final_predictions.append(most_common)

        # Calculate simple confidence as the proportion of models agreeing
        final_confidences = []
        for i in range(len(all_predictions[0])):
            predictor_votes = all_predictions[:, i]
            most_common = Counter(predictor_votes).most_common(1)[0]
            confidence = most_common[1] / len(predictor_votes)  # Proportion of agreement
            final_confidences.append(confidence)

        return np.array(final_predictions), np.array(final_confidences)

    def fit(self, df: pd.DataFrame):
        """Train the complete CEaSR model"""
        print("Training CEaSR model...")

        # Preprocess data
        df_processed = self.preprocess_data(df)

        # Encode target variable FIRST
        self.label_encoders['target'] = LabelEncoder()
        y_encoded = self.label_encoders['target'].fit_transform(df_processed['target_category'])

        # Build association strength models
        self.build_association_strength_models(df_processed)

        # Create weak predictors
        predictors = self.create_weak_predictors(df_processed)

        # Train confidence models
        self.train_confidence_models(predictors, y_encoded)

        print("Model training completed!")

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on new data"""
        print("Making predictions...")

        # Preprocess
        df_processed = self.preprocess_data(df)

        # Create predictors for new data
        predictors = self.create_weak_predictors(df_processed)

        # Make ensemble predictions
        predictions_encoded, confidences = self.ensemble_predict(predictors)

        # Decode predictions
        predictions = self.label_encoders['target'].inverse_transform(predictions_encoded.astype(int))

        return predictions, confidences

    def evaluate(self, df: pd.DataFrame) -> Dict:
        """Evaluate model performance"""
        print("Evaluating model...")

        # Get predictions
        y_pred, confidences = self.predict(df)

        # Get true labels
        df_processed = self.preprocess_data(df)
        y_true = df_processed['target_category']

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)

        # High confidence accuracy
        high_conf_mask = confidences > self.config['confidence_threshold']
        if high_conf_mask.sum() > 0:
            high_conf_accuracy = accuracy_score(
                y_true[high_conf_mask],
                y_pred[high_conf_mask]
            )
        else:
            high_conf_accuracy = 0.0

        results = {
            'overall_accuracy': accuracy,
            'high_confidence_accuracy': high_conf_accuracy,
            'high_confidence_coverage': high_conf_mask.mean(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }

        return results

    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'config': self.config,
            'label_encoders': self.label_encoders,
            'association_models': self.association_models,
            'confidence_models': self.confidence_models,
            'supplier_category_counts': dict(self.supplier_category_counts),
            'description_category_counts': dict(self.description_category_counts),
            'tower_category_counts': dict(self.tower_category_counts),
            'vectorizers': self.vectorizers,
            'unspsc_data': self.unspsc_data,
            'unspsc_mapping': self.unspsc_mapping
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.config = model_data['config']
        self.label_encoders = model_data['label_encoders']
        self.association_models = model_data['association_models']
        self.confidence_models = model_data['confidence_models']
        self.supplier_category_counts = defaultdict(Counter, model_data['supplier_category_counts'])
        self.description_category_counts = defaultdict(Counter, model_data['description_category_counts'])
        self.tower_category_counts = defaultdict(Counter, model_data.get('tower_category_counts', {}))
        self.vectorizers = model_data['vectorizers']
        self.unspsc_data = model_data.get('unspsc_data', None)
        self.unspsc_mapping = model_data.get('unspsc_mapping', {})

        print(f"Model loaded from {filepath}")


def main():
    """Main function to run the classification model"""
    print("Large Scale Personalized Classification of Financial Transactions")
    print("=" * 70)

    # Load data
    print("Loading data...")
    spend_data = pd.read_excel('/Users/sujoymukherjee/code/spendplatform/context/sample spend data- filled.xlsx')

    # Load UNSPSC data
    print("Loading UNSPSC data...")
    unspsc_data = pd.read_excel('/Users/sujoymukherjee/code/spendplatform/context/UNGM_UNSPSC_01-Sep-2025..xlsx')

    # Initialize classifier with UNSPSC data
    classifier = TransactionClassifier(unspsc_data=unspsc_data)

    # Split data for training and testing
    train_data, test_data = train_test_split(
        spend_data, test_size=0.2, random_state=42, stratify=spend_data['Tower (Practice)'].fillna('Unknown')
    )

    # Train model
    classifier.fit(train_data)

    # Evaluate model
    results = classifier.evaluate(test_data)

    print("\nEvaluation Results:")
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    print(f"High Confidence Accuracy: {results['high_confidence_accuracy']:.4f}")
    print(f"High Confidence Coverage: {results['high_confidence_coverage']:.4f}")

    # Save model
    classifier.save_model('/Users/sujoymukherjee/code/spendplatform/classificationmodel/cesar_model.pkl')

    print("\nModel training and evaluation completed!")


if __name__ == "__main__":
    main()
