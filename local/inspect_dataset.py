"""
Script to inspect CICIDS2017 dataset structure
Helps identify column names, label column, and data format
"""

import os
import pandas as pd
import config

def inspect_dataset(file_path):
    """Inspect a single dataset file"""
    print("\n" + "="*80)
    print(f"Inspecting: {os.path.basename(file_path)}")
    print("="*80)
    
    try:
        # Try different encodings
        try:
            df = pd.read_csv(file_path, encoding='utf-8', nrows=5)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin-1', nrows=5)
        
        print(f"\nShape: {df.shape[0]} rows (showing first 5), {df.shape[1]} columns")
        print(f"\nColumn names:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")
        
        # Look for label columns
        label_candidates = [col for col in df.columns 
                           if 'label' in col.lower() or 'attack' in col.lower() or 'class' in col.lower()]
        print(f"\nPossible label columns: {label_candidates}")
        
        # Check for BENIGN values
        for col in label_candidates:
            if col in df.columns:
                unique_vals = df[col].unique()
                print(f"\n  {col} unique values (first 5 rows): {unique_vals[:10]}")
                if 'BENIGN' in str(unique_vals):
                    print(f"    [OK] Contains 'BENIGN' - likely the label column!")
        
        # Show data types
        print(f"\nData types:")
        print(df.dtypes)
        
        # Show sample data
        print(f"\nFirst few rows:")
        print(df.head(2))
        
        return df.columns.tolist(), label_candidates
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None

def main():
    """Inspect all CSV files in data directory"""
    print("="*80)
    print("CICIDS2017 Dataset Inspector")
    print("="*80)
    
    data_dir = config.DATA_DIR
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return
    
    print(f"\nFound {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f}")
    
    # Inspect first file in detail
    if csv_files:
        first_file = os.path.join(data_dir, csv_files[0])
        columns, label_cols = inspect_dataset(first_file)
        
        # Check if we can read full file
        print("\n" + "="*80)
        print("Checking full file size...")
        print("="*80)
        try:
            try:
                df_full = pd.read_csv(first_file, encoding='utf-8')
            except UnicodeDecodeError:
                df_full = pd.read_csv(first_file, encoding='latin-1')
            
            print(f"Full file: {len(df_full)} rows, {len(df_full.columns)} columns")
            print(f"Memory usage: {df_full.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Check for missing values
            missing = df_full.isnull().sum()
            if missing.sum() > 0:
                print(f"\nColumns with missing values:")
                print(missing[missing > 0])
            else:
                print("\nNo missing values found")
            
            # Check label distribution if we found label column
            if label_cols:
                for col in label_cols:
                    if col in df_full.columns:
                        print(f"\nLabel distribution in '{col}':")
                        print(df_full[col].value_counts())
                        
        except Exception as e:
            print(f"Error reading full file: {e}")
    
    print("\n" + "="*80)
    print("Recommendations:")
    print("="*80)
    print("1. If multiple CSV files exist, they should be combined")
    print("2. Identify the correct label column name")
    print("3. Check if 'BENIGN' is used for normal traffic")
    print("4. Verify all files have the same column structure")

if __name__ == "__main__":
    main()

