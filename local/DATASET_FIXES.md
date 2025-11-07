# Dataset Issues Fixed

## Problems Identified

1. **Label Column Name**: The CICIDS2017 dataset uses `" Label"` (with a leading space) instead of `"Label"`
2. **Multiple Files**: The dataset consists of 8 separate CSV files that need to be combined
3. **Missing Values**: Some columns have missing values (e.g., "Flow Bytes/s" has 4 missing values)

## Fixes Applied

### 1. Enhanced Label Column Detection
- Updated `src/data_preprocessing.py` to check for:
  - `'Label'`, `'label'`, `'Label.1'`, `' Label'`, `' label'`
  - Any column containing 'label' (case-insensitive)
- Added robust label handling with whitespace trimming
- Case-insensitive comparison for 'BENIGN' values

### 2. Multiple File Support
- Modified `load_data()` to accept either a single file or a directory
- When a directory is provided, all CSV files are automatically loaded and combined
- Updated `main.py` to pass the data directory instead of a single file

### 3. Missing Value Handling
- Changed from simple `dropna()` to intelligent filling:
  - Numeric columns: filled with median
  - Categorical columns: filled with mode
- Only drops rows if filling fails

### 4. Better Diagnostics
- Added label distribution display
- Shows which label column is being used
- Displays anomaly rate after preprocessing

## Dataset Structure (CICIDS2017)

- **79 columns** of flow-based features
- **Label column**: `" Label"` (with leading space)
- **Label values**: `'BENIGN'` for normal traffic, various attack types for anomalies
- **8 CSV files** representing different days/attack scenarios:
  - Monday-WorkingHours.pcap_ISCX.csv
  - Tuesday-WorkingHours.pcap_ISCX.csv
  - Wednesday-workingHours.pcap_ISCX.csv
  - Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
  - Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
  - Friday-WorkingHours-Morning.pcap_ISCX.csv
  - Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
  - Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv

## Usage

The code now automatically:
1. Detects all CSV files in the `data/` directory
2. Combines them into a single dataset
3. Correctly identifies the label column (handles the space)
4. Handles missing values intelligently
5. Shows detailed preprocessing information

Run with:
```bash
python main.py
```

Or inspect the dataset first:
```bash
python inspect_dataset.py
```

