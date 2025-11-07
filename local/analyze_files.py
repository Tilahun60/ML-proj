"""
Analyze CICIDS2017 CSV files to recommend best file(s) for training
"""

import os
import pandas as pd
import numpy as np
import config

def analyze_file(file_path):
    """Analyze a single CSV file"""
    print(f"\n{'='*80}")
    print(f"Analyzing: {os.path.basename(file_path)}")
    print('='*80)
    
    try:
        # Try reading with different encodings
        try:
            df = pd.read_csv(file_path, encoding='utf-8', nrows=10000)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin-1', nrows=10000)
        
        # Get full size
        try:
            df_full = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df_full = pd.read_csv(file_path, encoding='latin-1')
        
        # Find label column
        label_col = None
        for col in [' Label', 'Label', 'label']:
            if col in df_full.columns:
                label_col = col
                break
        
        if label_col is None:
            print("  [WARNING] No label column found!")
            return None
        
        # Analyze labels
        label_counts = df_full[label_col].value_counts()
        total = len(df_full)
        benign_count = label_counts.get('BENIGN', 0)
        attack_count = total - benign_count
        
        print(f"  Total samples: {total:,}")
        print(f"  BENIGN: {benign_count:,} ({benign_count/total*100:.1f}%)")
        print(f"  Attacks: {attack_count:,} ({attack_count/total*100:.1f}%)")
        
        print(f"\n  Attack types:")
        for attack_type, count in label_counts.head(10).items():
            if attack_type != 'BENIGN':
                print(f"    - {attack_type}: {count:,}")
        
        # Estimate memory for sequences
        n_features = len([c for c in df_full.columns if c != label_col])
        seq_len = config.SEQUENCE_LENGTH
        n_sequences = total - seq_len + 1
        memory_gb = (n_sequences * seq_len * n_features * 8) / (1024**3)  # float64 = 8 bytes
        
        print(f"\n  Estimated memory (seq_len={seq_len}): {memory_gb:.2f} GB")
        
        # Check for issues
        issues = []
        if benign_count == 0:
            issues.append("No BENIGN traffic")
        if attack_count == 0:
            issues.append("No attack traffic")
        if memory_gb > 10:
            issues.append(f"High memory: {memory_gb:.2f}GB")
        
        if issues:
            print(f"\n  [WARNING] Issues: {', '.join(issues)}")
        else:
            print(f"\n  [OK] Good for training")
        
        return {
            'file': os.path.basename(file_path),
            'total': total,
            'benign': benign_count,
            'attacks': attack_count,
            'attack_rate': attack_count/total,
            'memory_gb': memory_gb,
            'issues': issues,
            'label_col': label_col
        }
        
    except Exception as e:
        print(f"  [ERROR] Error: {e}")
        return None

def main():
    """Analyze all CSV files and make recommendations"""
    print("="*80)
    print("CICIDS2017 File Analysis")
    print("="*80)
    
    data_dir = config.DATA_DIR
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    csv_files.sort()
    
    results = []
    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        result = analyze_file(file_path)
        if result:
            results.append(result)
    
    # Summary and recommendations
    print("\n" + "="*80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*80)
    
    print("\n[FILE COMPARISON]")
    print(f"{'File':<50} {'Samples':<12} {'BENIGN%':<10} {'Memory(GB)':<12} {'Status'}")
    print("-"*100)
    
    for r in results:
        status = "[OK] Good" if len(r['issues']) == 0 else "[WARNING] Issues"
        print(f"{r['file']:<50} {r['total']:<12,} {r['benign']/r['total']*100:<10.1f} {r['memory_gb']:<12.2f} {status}")
    
    # Best single file recommendation
    print("\n[RECOMMENDATIONS]")
    print("-"*80)
    
    # Find files with both BENIGN and attacks, reasonable size
    good_files = [r for r in results if r['benign'] > 0 and r['attacks'] > 0 and r['memory_gb'] < 5]
    
    if good_files:
        # Sort by attack rate (want ~15-25% attacks for good balance)
        good_files.sort(key=lambda x: abs(x['attack_rate'] - 0.20))
        best = good_files[0]
        print(f"\n1. BEST SINGLE FILE: {best['file']}")
        print(f"   - {best['total']:,} samples")
        print(f"   - {best['attack_rate']*100:.1f}% attack rate (good balance)")
        print(f"   - {best['memory_gb']:.2f} GB memory needed")
        print(f"   - Has both BENIGN and attack traffic")
    
    # Alternative: smaller files
    small_files = [r for r in results if r['memory_gb'] < 2 and r['benign'] > 0 and r['attacks'] > 0]
    if small_files:
        small_files.sort(key=lambda x: x['memory_gb'])
        print(f"\n2. BEST FOR LIMITED MEMORY: {small_files[0]['file']}")
        print(f"   - {small_files[0]['total']:,} samples")
        print(f"   - {small_files[0]['memory_gb']:.2f} GB memory needed")
    
    # Combination recommendation
    print(f"\n3. RECOMMENDED COMBINATION (2-3 files):")
    print(f"   Combine files with different attack types for better model generalization")
    print(f"   Suggested: Tuesday + one attack-specific file (e.g., PortScan or DDoS)")
    
    print("\n" + "="*80)
    print("To use a specific file, edit config.py:")
    print("  SELECTED_FILES = ['Tuesday-WorkingHours.pcap_ISCX.csv']")
    print("="*80)

if __name__ == "__main__":
    main()

