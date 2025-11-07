# Step-by-Step Guide: Uploading to Google Colab

## 📋 Prerequisites
- Google account
- CICIDS2017 dataset CSV file (or use the one provided)

---

## 🚀 Step 1: Open Google Colab

1. Go to **https://colab.research.google.com/**
2. Sign in with your Google account
3. Click **"New notebook"** or **"File → New notebook"**

---

## ⚙️ Step 2: Enable GPU (IMPORTANT!)

**This is essential for fast training (1.5-2 hours vs 4-6 hours)**

1. Click **"Runtime"** in the top menu
2. Select **"Change runtime type"**
3. In the popup:
   - **Hardware accelerator**: Select **"GPU"**
   - **GPU type**: Choose **"T4"** (free) or **"V100"** (Pro)
4. Click **"Save"**

✅ **Verify GPU is enabled** (run in a cell):
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## 📦 Step 3: Install Dependencies

Create a new cell and run:

```python
%pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
%pip install numpy pandas scikit-learn matplotlib seaborn tqdm scipy
```

⏱️ **Time**: ~2-3 minutes

---

## 📁 Step 4: Create Project Structure

Run this in a new cell:

```python
import os

# Create project directories
os.makedirs('/content/ML-proj/data', exist_ok=True)
os.makedirs('/content/ML-proj/models', exist_ok=True)
os.makedirs('/content/ML-proj/results', exist_ok=True)
os.makedirs('/content/ML-proj/src', exist_ok=True)

print("✓ Directories created!")
```

---

## 📤 Step 5: Upload Project Files

You have **3 options** to upload files:

### **Option A: Upload via Colab UI (Easiest)**

1. Click the **folder icon** (📁) in the left sidebar
2. Click **"Upload"** button
3. Upload these files from your `colab/` folder:
   - `main.py`
   - `config.py`
   - `requirements.txt`
   - All files from `src/` folder:
     - `src/data_preprocessing.py`
     - `src/transformer_model.py`
     - `src/lstm_model.py`
     - `src/random_forest_model.py`
     - `src/train.py`
     - `src/evaluate.py`
     - `src/visualize.py`
     - `src/__init__.py`

4. After upload, **move files to correct locations**:
```python
import shutil
import os

# Move files to project directory
os.makedirs('/content/ML-proj', exist_ok=True)
os.makedirs('/content/ML-proj/src', exist_ok=True)

# Move main files
if os.path.exists('/content/main.py'):
    shutil.move('/content/main.py', '/content/ML-proj/')
if os.path.exists('/content/config.py'):
    shutil.move('/content/config.py', '/content/ML-proj/')

# Move src files
for file in ['data_preprocessing.py', 'transformer_model.py', 'lstm_model.py', 
              'random_forest_model.py', 'train.py', 'evaluate.py', 'visualize.py', '__init__.py']:
    if os.path.exists(f'/content/{file}'):
        shutil.move(f'/content/{file}', '/content/ML-proj/src/')

print("✓ Files moved to correct locations!")
```

### **Option B: Use Google Drive (Recommended for large files)**

1. **Upload to Google Drive first:**
   - Upload your `colab/` folder to Google Drive
   - Note the path (e.g., `MyDrive/ML-proj/colab/`)

2. **Mount Drive in Colab:**
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy files from Drive
import shutil
import os

drive_path = '/content/drive/MyDrive/ML-proj/colab'  # Adjust path as needed
target_path = '/content/ML-proj'

# Copy all files
if os.path.exists(drive_path):
    shutil.copytree(drive_path, target_path, dirs_exist_ok=True)
    print("✓ Files copied from Drive!")
else:
    print(f"⚠️ Path not found: {drive_path}")
    print("Please adjust the drive_path variable")
```

### **Option C: Use GitHub (If you have a repo)**

```python
!git clone https://github.com/yourusername/ML-proj.git /content/ML-proj
# Then copy colab folder contents
```

---

## 📊 Step 6: Upload Dataset

Upload the CICIDS2017 CSV file:

### **Option 1: Direct Upload**
1. Click folder icon (📁) in left sidebar
2. Click **"Upload"**
3. Upload: `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
4. Move to data folder:
```python
import shutil
if os.path.exists('/content/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'):
    shutil.move('/content/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 
                '/content/ML-proj/data/')
    print("✓ Dataset uploaded!")
```

### **Option 2: From Google Drive**
```python
# If dataset is in Drive
drive_path = '/content/drive/MyDrive/CICIDS2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
if os.path.exists(drive_path):
    shutil.copy(drive_path, '/content/ML-proj/data/')
    print("✓ Dataset copied from Drive!")
```

### **Option 3: Download from URL (if you have a direct link)**
```python
!wget -P /content/ML-proj/data/ <your_dataset_url>
```

---

## ✅ Step 7: Verify Files Are in Place

Run this to check:

```python
import os

print("Checking project structure...\n")

# Check main files
files_to_check = [
    '/content/ML-proj/main.py',
    '/content/ML-proj/config.py',
]

for file in files_to_check:
    status = "✓" if os.path.exists(file) else "✗"
    print(f"{status} {file}")

# Check src files
print("\nChecking src/ files:")
src_files = ['data_preprocessing.py', 'transformer_model.py', 'lstm_model.py',
             'random_forest_model.py', 'train.py', 'evaluate.py', 'visualize.py']
for file in src_files:
    path = f'/content/ML-proj/src/{file}'
    status = "✓" if os.path.exists(path) else "✗"
    print(f"{status} {path}")

# Check dataset
dataset_path = '/content/ML-proj/data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
if os.path.exists(dataset_path):
    size = os.path.getsize(dataset_path) / (1024**2)
    print(f"\n✓ Dataset found: {size:.1f} MB")
else:
    print("\n✗ Dataset not found!")

print("\n" + "="*50)
print("If all files show ✓, you're ready to run!")
```

---

## 🏃 Step 8: Update Paths and Run

```python
import sys
sys.path.append('/content/ML-proj')

# Update config paths
import config
config.DATA_DIR = '/content/ML-proj/data'
config.MODEL_DIR = '/content/ML-proj/models'
config.RESULTS_DIR = '/content/ML-proj/results'

print("✓ Paths configured!")
print(f"Data directory: {config.DATA_DIR}")
print(f"Model directory: {config.MODEL_DIR}")
print(f"Results directory: {config.RESULTS_DIR}")
```

---

## 🚀 Step 9: Run Training

```python
from main import main
import time

start_time = time.time()
print("Starting training...")
print("="*80)

main()

elapsed = time.time() - start_time
print("\n" + "="*80)
print(f"✓ Training completed in {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
print("="*80)
```

⏱️ **Expected time**: 1.5-2 hours with GPU

---

## 💾 Step 10: Download Results

After training completes:

```python
from google.colab import files
import os

# Download visualizations
results_dir = '/content/ML-proj/results'
if os.path.exists(results_dir):
    print("Downloading visualizations...")
    for file in os.listdir(results_dir):
        if file.endswith('.png'):
            filepath = os.path.join(results_dir, file)
            files.download(filepath)
            print(f"  ✓ Downloaded: {file}")

# Download models (optional - they're large)
models_dir = '/content/ML-proj/models'
if os.path.exists(models_dir):
    print("\nDownloading models...")
    for file in os.listdir(models_dir):
        if file.endswith('.pth') or file.endswith('.pkl'):
            filepath = os.path.join(models_dir, file)
            files.download(filepath)
            print(f"  ✓ Downloaded: {file}")

print("\n✓ All files downloaded!")
```

---

## 📝 Quick Checklist

Before running, make sure:

- [ ] GPU is enabled (Runtime → Change runtime type → GPU)
- [ ] Dependencies installed
- [ ] All project files uploaded to `/content/ML-proj/`
- [ ] Dataset CSV file in `/content/ML-proj/data/`
- [ ] Paths updated in config
- [ ] Ready to run!

---

## 🆘 Troubleshooting

### "Module not found" error
- Make sure you added `/content/ML-proj` to `sys.path`
- Verify all files are in correct locations

### "File not found" error
- Check file paths using the verification step
- Make sure dataset is in `/content/ML-proj/data/`

### Out of Memory (OOM)
- Reduce `batch_size` in `config.py` to 64
- Set `MAX_SAMPLES = 100000` in `config.py`

### Session Disconnected
- Colab disconnects after 90 min of inactivity
- Add periodic print statements to keep session alive
- Consider Colab Pro for longer sessions

---

## 🎯 Alternative: Use the Notebook

Instead of manual upload, you can use the ready-made notebook:

1. Upload `Network_Anomaly_Detection_Colab.ipynb` to Colab
2. Run all cells sequentially
3. The notebook handles everything automatically!

---

## 📚 Next Steps

After successful upload:
- See `README.md` for usage instructions
- See `COLAB_GUIDE.md` for detailed Colab tips
- Check results in `/content/ML-proj/results/`

Good luck! 🚀

