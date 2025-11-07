"""
Data preprocessing module for CICIDS2017 dataset
Handles loading, cleaning, normalization, and time-series window creation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Preprocess network log data for anomaly detection"""
    
    def __init__(self, sequence_length=100, normalize=True, selected_files=None):
        """
        Initialize preprocessor
        
        Args:
            sequence_length: Length of time-series windows
            normalize: Whether to normalize features
            selected_files: List of specific CSV filenames to use (None = use all)
        """
        self.sequence_length = sequence_length
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.class_weights = None
        self.selected_files = selected_files
        
    def load_data(self, file_path):
        """
        Load dataset from CSV file or directory of CSV files
        
        Args:
            file_path: Path to CSV file or directory containing CSV files
            
        Returns:
            DataFrame with loaded data
        """
        import os
        
        # Check if it's a directory or single file
        if os.path.isdir(file_path):
            # Load CSV files from directory
            all_csv_files = [f for f in os.listdir(file_path) if f.endswith('.csv')]
            
            # Filter by selected files if specified
            if hasattr(self, 'selected_files') and self.selected_files is not None:
                csv_files = [f for f in all_csv_files if f in self.selected_files]
                if not csv_files:
                    raise ValueError(f"Selected files not found. Available: {all_csv_files}")
                print(f"Using {len(csv_files)} selected file(s) out of {len(all_csv_files)} available")
            else:
                csv_files = all_csv_files
            
            if not csv_files:
                raise ValueError(f"No CSV files found in directory: {file_path}")
            
            print(f"Loading {len(csv_files)} CSV file(s)...")
            dfs = []
            for csv_file in csv_files:
                file_path_full = os.path.join(file_path, csv_file)
                print(f"  Loading {csv_file}...")
                try:
                    df_temp = pd.read_csv(file_path_full, encoding='utf-8', low_memory=False)
                except UnicodeDecodeError:
                    df_temp = pd.read_csv(file_path_full, encoding='latin-1', low_memory=False)
                dfs.append(df_temp)
                print(f"    Loaded {len(df_temp):,} rows")
            
            # Combine all dataframes
            df = pd.concat(dfs, ignore_index=True)
            print(f"\nCombined dataset: {len(df):,} samples with {len(df.columns)} features")
        else:
            # Single file
            print(f"Loading data from {file_path}...")
            try:
                df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin-1', low_memory=False)
            
            print(f"Loaded {len(df)} samples with {len(df.columns)} features")
        
        return df
    
    def clean_data(self, df):
        """
        Clean and prepare data
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        print("Cleaning data...")
        
        # Handle infinity and extremely large values first
        initial_len = len(df)
        print("Checking for infinity and extreme values...")
        inf_count = 0
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                # Check for infinity
                inf_mask = np.isinf(df[col])
                if inf_mask.sum() > 0:
                    inf_count += inf_mask.sum()
                    # Replace infinity with NaN (will be handled below)
                    df.loc[inf_mask, col] = np.nan
                    print(f"  Found {inf_mask.sum()} infinity values in '{col}'")
                
                # Check for extremely large values (larger than float64 max)
                max_float = np.finfo(np.float64).max
                large_mask = np.abs(df[col]) > max_float * 0.9
                if large_mask.sum() > 0:
                    # Replace with a reasonable large value
                    df.loc[large_mask, col] = np.sign(df.loc[large_mask, col]) * max_float * 0.9
                    print(f"  Found {large_mask.sum()} extremely large values in '{col}'")
        
        if inf_count > 0:
            print(f"Total infinity values found: {inf_count}")
        
        # Handle missing values - fill numeric columns with median, drop if too many missing
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"Found missing values in {missing_counts[missing_counts > 0].shape[0]} columns")
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if df[col].dtype in ['int64', 'float64']:
                        # Fill numeric columns with median
                        median_val = df[col].median()
                        if pd.isna(median_val):
                            # If median is also NaN, use 0
                            median_val = 0
                        df[col].fillna(median_val, inplace=True)
                        print(f"  Filled '{col}' with median: {median_val}")
                    else:
                        # For categorical, fill with mode or drop
                        mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                        df[col].fillna(mode_val, inplace=True)
                        print(f"  Filled '{col}' with mode: {mode_val}")
        
        # Remove any remaining rows with missing values (should be none after filling)
        df = df.dropna()
        print(f"Final dataset after cleaning: {len(df)} rows (removed {initial_len - len(df)} rows)")
        
        # Identify label column (CICIDS2017 uses ' Label' with leading space)
        label_col = None
        # Check for common label column names (including with spaces)
        label_candidates = ['Label', 'label', 'Label.1', ' Label', ' label']
        for col in label_candidates:
            if col in df.columns:
                label_col = col
                break
        
        # Also check for columns containing 'label' (case insensitive)
        if label_col is None:
            for col in df.columns:
                if 'label' in col.lower():
                    label_col = col
                    print(f"Found label column: '{col}'")
                    break
        
        if label_col is None:
            raise ValueError(f"Label column not found. Searched for: {label_candidates}")
        
        print(f"Using label column: '{label_col}'")
        
        # Convert label to binary: BENIGN = 0, others = 1 (anomaly)
        # Handle potential whitespace issues
        df[label_col] = df[label_col].astype(str).str.strip()
        df['is_anomaly'] = (df[label_col].str.upper() != 'BENIGN').astype(int)
        
        # Show label distribution
        print(f"\nLabel distribution:")
        label_counts = df[label_col].value_counts()
        print(label_counts.head(10))
        print(f"\nAnomaly rate: {df['is_anomaly'].mean():.2%}")
        
        # Remove label columns from features
        feature_cols = [col for col in df.columns 
                       if col not in [label_col, 'is_anomaly']]
        
        # Remove non-numeric columns (or encode them)
        numeric_cols = []
        categorical_cols = []
        
        for col in feature_cols:
            if df[col].dtype in ['int64', 'float64']:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        
        if categorical_cols:
            print(f"Encoding {len(categorical_cols)} categorical features...")
            for col in categorical_cols:
                df[col] = pd.Categorical(df[col]).codes
        
        self.feature_columns = numeric_cols + categorical_cols
        
        # Select only numeric features
        X = df[self.feature_columns].values
        y = df['is_anomaly'].values
        
        print(f"Final dataset: {len(X)} samples, {X.shape[1]} features")
        print(f"Anomaly rate: {y.mean():.2%}")
        
        return X, y
    
    def normalize_features(self, X_train, X_val=None, X_test=None):
        """
        Normalize features using StandardScaler
        
        Args:
            X_train: Training features
            X_val: Validation features (optional)
            X_test: Test features (optional)
            
        Returns:
            Normalized feature arrays
        """
        if not self.normalize:
            return X_train, X_val, X_test
        
        print("Normalizing features...")
        
        # Final check for infinity and NaN values before normalization
        # Use a robust approach: replace inf/NaN and clip extreme values
        if np.any(np.isinf(X_train)) or np.any(np.isnan(X_train)):
            print("Warning: Found infinity or NaN in training data. Replacing...")
            # Calculate reasonable clipping values based on finite data
            finite_mask = np.isfinite(X_train)
            if np.any(finite_mask):
                finite_data = X_train[finite_mask]
                # Use 99.9th percentile as max value for clipping
                max_abs_val = np.percentile(np.abs(finite_data), 99.9)
                clip_val = max(max_abs_val * 10, 1e6)  # At least 1e6
            else:
                clip_val = 1e6
            
            # Replace inf/NaN and clip
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=clip_val, neginf=-clip_val)
            X_train = np.clip(X_train, -clip_val, clip_val)
            print(f"  Clipped values to range [-{clip_val:.2e}, {clip_val:.2e}]")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        X_val_scaled = None
        if X_val is not None:
            # Check for infinity/NaN in validation set
            if np.any(np.isinf(X_val)) or np.any(np.isnan(X_val)):
                print("Warning: Found infinity or NaN in validation data. Replacing...")
                # Use same clipping value as training
                finite_mask = np.isfinite(X_train)  # Use training data for clip value
                if np.any(finite_mask):
                    finite_data = X_train[finite_mask]
                    max_abs_val = np.percentile(np.abs(finite_data), 99.9)
                    clip_val = max(max_abs_val * 10, 1e6)
                else:
                    clip_val = 1e6
                X_val = np.nan_to_num(X_val, nan=0.0, posinf=clip_val, neginf=-clip_val)
                X_val = np.clip(X_val, -clip_val, clip_val)
            X_val_scaled = self.scaler.transform(X_val)
        
        X_test_scaled = None
        if X_test is not None:
            # Check for infinity/NaN in test set
            if np.any(np.isinf(X_test)) or np.any(np.isnan(X_test)):
                print("Warning: Found infinity or NaN in test data. Replacing...")
                # Use same clipping value as training
                finite_mask = np.isfinite(X_train)  # Use training data for clip value
                if np.any(finite_mask):
                    finite_data = X_train[finite_mask]
                    max_abs_val = np.percentile(np.abs(finite_data), 99.9)
                    clip_val = max(max_abs_val * 10, 1e6)
                else:
                    clip_val = 1e6
                X_test = np.nan_to_num(X_test, nan=0.0, posinf=clip_val, neginf=-clip_val)
                X_test = np.clip(X_test, -clip_val, clip_val)
            X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def create_sequences(self, X, y):
        """
        Create time-series sequences from data
        
        Args:
            X: Feature array (n_samples, n_features)
            y: Label array (n_samples,)
            
        Returns:
            X_seq: Sequence array (n_sequences, sequence_length, n_features)
            y_seq: Label array (n_sequences,)
        """
        print(f"Creating sequences with length {self.sequence_length}...")
        
        n_samples, n_features = X.shape
        n_sequences = n_samples - self.sequence_length + 1
        
        if n_sequences <= 0:
            raise ValueError(f"Not enough samples to create sequences. Need at least {self.sequence_length} samples")
        
        X_seq = np.zeros((n_sequences, self.sequence_length, n_features))
        y_seq = np.zeros(n_sequences, dtype=int)
        
        for i in range(n_sequences):
            X_seq[i] = X[i:i+self.sequence_length]
            # Use the label of the last element in the sequence
            y_seq[i] = y[i+self.sequence_length-1]
        
        print(f"Created {n_sequences} sequences")
        print(f"Anomaly rate in sequences: {y_seq.mean():.2%}")
        
        return X_seq, y_seq
    
    def compute_class_weights(self, y):
        """
        Compute class weights for imbalanced dataset
        
        Args:
            y: Label array
            
        Returns:
            Dictionary of class weights
        """
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        self.class_weights = dict(zip(classes, weights))
        print(f"Class weights: {self.class_weights}")
        return self.class_weights
    
    def preprocess_pipeline(self, file_path, create_sequences=True, max_samples=None):
        """
        Complete preprocessing pipeline
        
        Args:
            file_path: Path to dataset CSV
            create_sequences: Whether to create time-series sequences
            max_samples: Maximum number of samples to use (None = use all)
            
        Returns:
            Preprocessed data splits
        """
        # Load and clean
        df = self.load_data(file_path)
        X, y = self.clean_data(df)
        
        # Sample data if max_samples is specified (for memory efficiency)
        if max_samples is not None and len(X) > max_samples:
            print(f"\nSampling {max_samples} samples from {len(X)} total samples...")
            # Use stratified sampling to maintain class distribution
            _, X, _, y = train_test_split(
                X, y, 
                train_size=max_samples, 
                random_state=42, 
                stratify=y
            )
            print(f"Using {len(X)} samples after stratified sampling")
            print(f"Anomaly rate after sampling: {y.mean():.2%}")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2/0.9, random_state=42, stratify=y_temp
        )
        
        print(f"\nData splits:")
        print(f"Train: {len(X_train)} samples")
        print(f"Validation: {len(X_val)} samples")
        print(f"Test: {len(X_test)} samples")
        
        # Normalize
        X_train, X_val, X_test = self.normalize_features(X_train, X_val, X_test)
        
        # Compute class weights
        self.compute_class_weights(y_train)
        
        # Create sequences if needed
        if create_sequences:
            X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
            X_val_seq, y_val_seq = self.create_sequences(X_val, y_val)
            X_test_seq, y_test_seq = self.create_sequences(X_test, y_test)
            
            return {
                'train': (X_train_seq, y_train_seq),
                'val': (X_val_seq, y_val_seq),
                'test': (X_test_seq, y_test_seq),
                'n_features': X_train.shape[1],
                'class_weights': self.class_weights
            }
        else:
            return {
                'train': (X_train, y_train),
                'val': (X_val, y_val),
                'test': (X_test, y_test),
                'n_features': X_train.shape[1],
                'class_weights': self.class_weights
            }

