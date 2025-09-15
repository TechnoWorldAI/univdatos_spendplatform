"""
Data preprocessing utilities for the transaction classification system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import re
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class TransactionDataPreprocessor:
    """Preprocessor for transaction data"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.text_cleaners = {
            'remove_special_chars': self._remove_special_chars,
            'normalize_whitespace': self._normalize_whitespace,
            'lowercase': self._lowercase
        }

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main preprocessing pipeline"""
        print("Starting data preprocessing...")

        df = df.copy()

        # Handle missing values
        df = self._handle_missing_values(df)

        # Clean text fields
        df = self._clean_text_fields(df)

        # Extract features
        df = self._extract_features(df)

        # Normalize numerical features
        df = self._normalize_features(df)

        print(f"Preprocessing completed. Shape: {df.shape}")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Fill text fields with empty strings
        text_columns = ['INV_ITEM_DESC', 'SUPPLIER_NAME', 'Material Item Name',
                       'Order description', 'SUPPLIER_NO', 'Material Code']

        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('')

        # Fill numerical fields with 0
        numeric_columns = ['Item Invoice Value', 'PO Price', 'Unit Price',
                          'Item Qty Invoiced', 'PO purchase qty']

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Handle categorical fields
        categorical_columns = ['BU_CODE', 'BU _NAME', 'Region', 'Tower (Practice)']

        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')

        return df

    def _clean_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize text fields"""
        text_columns = ['INV_ITEM_DESC', 'SUPPLIER_NAME', 'Material Item Name', 'Order description']

        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(self._clean_text)

        return df

    def _clean_text(self, text: str) -> str:
        """Clean individual text strings"""
        if not isinstance(text, str) or text == '':
            return ''

        # Apply cleaning functions
        text = self._remove_special_chars(text)
        text = self._normalize_whitespace(text)
        text = self._lowercase(text)

        return text.strip()

    def _remove_special_chars(self, text: str) -> str:
        """Remove special characters"""
        # Keep alphanumeric, spaces, and basic punctuation
        text = re.sub(r'[^\w\s\-&\.]', ' ', text)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace"""
        text = re.sub(r'\s+', ' ', text)
        return text

    def _lowercase(self, text: str) -> str:
        """Convert to lowercase"""
        return text.lower()

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract additional features from the data"""

        # Extract supplier features
        if 'SUPPLIER_NAME' in df.columns:
            df['supplier_length'] = df['SUPPLIER_NAME'].str.len()
            df['supplier_word_count'] = df['SUPPLIER_NAME'].str.split().str.len()

        # Extract description features
        if 'INV_ITEM_DESC' in df.columns:
            df['desc_length'] = df['INV_ITEM_DESC'].str.len()
            df['desc_word_count'] = df['INV_ITEM_DESC'].str.split().str.len()

            # Check for specific keywords
            df['has_oil'] = df['INV_ITEM_DESC'].str.contains(r'\boil\b', case=False, regex=True).astype(int)
            df['has_diesel'] = df['INV_ITEM_DESC'].str.contains(r'\bdiesel\b', case=False, regex=True).astype(int)
            df['has_paint'] = df['INV_ITEM_DESC'].str.contains(r'\bpaint\b', case=False, regex=True).astype(int)
            df['has_epoxy'] = df['INV_ITEM_DESC'].str.contains(r'\bepoxy\b', case=False, regex=True).astype(int)

        # Extract amount-based features
        if 'Item Invoice Value' in df.columns:
            df['amount_log'] = np.log1p(df['Item Invoice Value'].abs())
            df['amount_category'] = pd.cut(df['Item Invoice Value'].abs(),
                                         bins=[0, 100, 1000, 10000, float('inf')],
                                         labels=['small', 'medium', 'large', 'xlarge'])

        # Extract date features
        date_columns = ['PO Creation Date', 'Invoice Date']
        for col in date_columns:
            if col in df.columns:
                try:
                    df[f'{col}_month'] = pd.to_datetime(df[col], errors='coerce').dt.month
                    df[f'{col}_year'] = pd.to_datetime(df[col], errors='coerce').dt.year
                except:
                    df[f'{col}_month'] = np.nan
                    df[f'{col}_year'] = np.nan

        return df

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features"""
        numeric_cols = ['supplier_length', 'supplier_word_count', 'desc_length',
                       'desc_word_count', 'amount_log']

        existing_numeric_cols = [col for col in numeric_cols if col in df.columns]

        if existing_numeric_cols:
            df[existing_numeric_cols] = self.scaler.fit_transform(df[existing_numeric_cols])

        return df

    def get_feature_importance_analysis(self, df: pd.DataFrame) -> Dict:
        """Analyze feature importance and correlations"""
        analysis = {}

        # Basic statistics
        analysis['shape'] = df.shape
        analysis['missing_values'] = df.isnull().sum().to_dict()

        # Categorical distributions
        categorical_cols = ['BU_CODE', 'BU _NAME', 'Region', 'Tower (Practice)']
        analysis['categorical_distributions'] = {}

        for col in categorical_cols:
            if col in df.columns:
                analysis['categorical_distributions'][col] = df[col].value_counts().to_dict()

        # Numerical statistics
        numeric_cols = ['Item Invoice Value', 'PO Price', 'Unit Price']
        analysis['numerical_stats'] = {}

        for col in numeric_cols:
            if col in df.columns:
                analysis['numerical_stats'][col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                }

        return analysis


class FeatureEngineer:
    """Feature engineering for transaction classification"""

    def __init__(self):
        self.feature_cache = {}

    def create_association_features(self, df: pd.DataFrame,
                                  association_dict: Dict,
                                  feature_prefix: str) -> pd.DataFrame:
        """Create association-based features"""
        features = []

        for idx, row in df.iterrows():
            feature_row = {}

            # Get the key for association lookup
            key = self._get_association_key(row, feature_prefix)

            if key in association_dict:
                associations = association_dict[key]
                total = sum(associations.values())

                # Calculate association strengths
                for category, count in associations.items():
                    strength = (count + 1) / (total + len(associations))
                    feature_row[f'{feature_prefix}_{category}'] = strength
            else:
                # Default values for unknown keys
                for category in df.get('target_category', pd.Series()).unique():
                    feature_row[f'{feature_prefix}_{category}'] = 1.0 / len(df.get('target_category', pd.Series()).unique())

            features.append(feature_row)

        return pd.DataFrame(features)

    def _get_association_key(self, row: pd.Series, feature_prefix: str) -> str:
        """Get the key for association lookup"""
        if feature_prefix == 'supplier':
            return row.get('SUPPLIER_NAME', '')
        elif feature_prefix == 'description':
            return row.get('INV_ITEM_DESC', '')
        elif feature_prefix == 'material':
            return row.get('Material Item Name', '')
        else:
            return ''

    def create_text_similarity_features(self, df: pd.DataFrame,
                                      reference_texts: List[str]) -> pd.DataFrame:
        """Create text similarity features"""
        features = []

        for idx, row in df.iterrows():
            feature_row = {}
            text = row.get('INV_ITEM_DESC', '') + ' ' + row.get('SUPPLIER_NAME', '')

            for i, ref_text in enumerate(reference_texts):
                similarity = self._calculate_text_similarity(text, ref_text)
                feature_row[f'text_sim_{i}'] = similarity

            features.append(feature_row)

        return pd.DataFrame(features)

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        if not text1 or not text2:
            return 0.0

        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features"""
        features = []

        for idx, row in df.iterrows():
            feature_row = {}

            # Date-based features
            for date_col in ['PO Creation Date', 'Invoice Date']:
                if date_col in row and pd.notna(row[date_col]):
                    try:
                        date = pd.to_datetime(row[date_col])
                        feature_row[f'{date_col}_month'] = date.month
                        feature_row[f'{date_col}_quarter'] = (date.month - 1) // 3 + 1
                        feature_row[f'{date_col}_day_of_week'] = date.dayofweek
                    except:
                        feature_row[f'{date_col}_month'] = 0
                        feature_row[f'{date_col}_quarter'] = 0
                        feature_row[f'{date_col}_day_of_week'] = 0

            features.append(feature_row)

        return pd.DataFrame(features)
