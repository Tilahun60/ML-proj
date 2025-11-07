# File Selection Recommendations for CICIDS2017

## Analysis Results

Based on the analysis of all CSV files, here are the recommendations:

## Best Single Files for Anomaly Detection

### 1. **Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv** ⭐ RECOMMENDED
- **Samples**: 225,745
- **Memory**: 6.56 GB
- **BENIGN**: 43.3% (97,718)
- **Attacks**: 56.7% (128,027 DDoS attacks)
- **Why**: Well-balanced dataset with good attack representation, manageable memory

### 2. **Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv**
- **Samples**: 286,467
- **Memory**: 8.32 GB
- **BENIGN**: 44.5% (127,537)
- **Attacks**: 55.5% (158,930 PortScan attacks)
- **Why**: Good balance, slightly larger but still manageable

### 3. **Wednesday-workingHours.pcap_ISCX.csv** (if you have >20GB RAM)
- **Samples**: 692,703
- **Memory**: 20.13 GB
- **BENIGN**: 63.5% (440,031)
- **Attacks**: 36.5% (multiple types: DoS Hulk, GoldenEye, slowloris, etc.)
- **Why**: Best for diversity - has multiple attack types, but requires more memory

## Files to Avoid

- **Monday-WorkingHours.pcap_ISCX.csv**: No attacks (100% BENIGN)
- **Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv**: Only 36 attacks (0.01%)
- **Friday-WorkingHours-Morning.pcap_ISCX.csv**: Only 1% attacks (mostly BENIGN)

## How to Use a Specific File

Edit `config.py`:

```python
# Use single best file
SELECTED_FILES = ["Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"]

# Or use multiple files for more diversity
SELECTED_FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
]

# Or use all files (default)
SELECTED_FILES = None
```

## Memory Requirements Summary

| Configuration | Memory Needed |
|--------------|---------------|
| Single DDoS file | ~6.5 GB |
| Single PortScan file | ~8.3 GB |
| DDoS + PortScan | ~15 GB |
| Wednesday (multiple attacks) | ~20 GB |
| All files combined | ~55 GB |

## Recommendation

**For most users**: Use **Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv**
- Good balance of normal and attack traffic
- Manageable memory (~6.5 GB)
- Single attack type is fine for anomaly detection (BENIGN vs Attack)

**For better generalization**: Combine 2-3 files with different attack types:
```python
SELECTED_FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
]
```

