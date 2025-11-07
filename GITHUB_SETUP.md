# GitHub Setup Guide

## Step-by-Step Instructions to Push to GitHub

### Step 1: Create GitHub Repository

1. Go to **https://github.com/**
2. Sign in to your account
3. Click the **"+"** icon in the top right
4. Select **"New repository"**
5. Fill in:
   - **Repository name**: `ML-proj` (or your preferred name)
   - **Description**: "Network Log Anomaly Detection using Transformer on Time-Series Data"
   - **Visibility**: Public or Private (your choice)
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click **"Create repository"**

### Step 2: Push from Local Machine

Run these commands in your terminal (in the ML-proj directory):

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Network Log Anomaly Detection project with local and colab versions"

# Add remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/ML-proj.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 3: Verify

1. Go to your GitHub repository page
2. You should see:
   - `README.md`
   - `local/` folder
   - `colab/` folder
   - `.gitignore`

## Alternative: Using GitHub CLI

If you have GitHub CLI installed:

```bash
gh repo create ML-proj --public --source=. --remote=origin --push
```

## What Gets Pushed

✅ **Will be pushed:**
- All source code (`local/src/`, `colab/src/`)
- Configuration files (`config.py`, `main.py`)
- Documentation (`README.md`, guides)
- Project structure

❌ **Will NOT be pushed** (due to .gitignore):
- Data files (`*.csv` in data folders)
- Trained models (`*.pth`, `*.pkl`)
- Results (`*.png`, `*.txt` in results folders)
- Python cache (`__pycache__/`)
- Virtual environments

## Repository Structure on GitHub

```
ML-proj/
├── .gitignore
├── README.md
├── local/
│   ├── main.py
│   ├── config.py
│   ├── README.md
│   ├── src/
│   └── ...
└── colab/
    ├── main.py
    ├── config.py
    ├── README.md
    ├── Network_Anomaly_Detection_Colab.ipynb
    ├── UPLOAD_TO_COLAB.md
    └── src/
```

## Future Updates

To push future changes:

```bash
git add .
git commit -m "Description of changes"
git push
```

## Troubleshooting

### "Repository not found"
- Check the repository URL is correct
- Make sure you have access to the repository

### "Authentication failed"
- Use GitHub Personal Access Token instead of password
- Or use SSH: `git remote set-url origin git@github.com:USERNAME/ML-proj.git`

### "Large files error"
- The .gitignore should prevent large files
- If issues persist, check file sizes: `git ls-files | xargs ls -lh`

## Security Note

Make sure `.gitignore` is working correctly - you don't want to accidentally push:
- Large dataset files
- API keys or secrets
- Personal data

The provided `.gitignore` should handle this, but always verify before pushing!

