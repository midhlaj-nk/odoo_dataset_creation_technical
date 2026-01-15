# How to Generate the Odoo Dataset using Kaggle (One-Click Setup)

This guide helps you run the professional dataset generation job in the background using Kaggle's free GPU instances and Groq AI speed.

## Step 1: Upload to Kaggle
1. Go to [kaggle.com](https://www.kaggle.com/) and Log in.
2. Click **Create** > **New Notebook**.
3. In the notebook editor, go to **File** > **Import Notebook**.
4. Upload the **`Run_on_Kaggle.ipynb`** file from this repository.

## Step 2: Configure Settings (Right Panel)
Before running, ensure these settings are active in the right-hand panel:
1. **Internet**: Switch to **On** (Requires phone verification if your account is new).
2. **Accelerator**: Select **GPU T4 x2**.

## Step 3: Run in Background (Vacation Mode)
To ensure the script runs for the full 12 hours even if you close your computer:
1. Do **NOT** just press the blue "Run All" button in the editor.
2. Click the **"Save Version"** button in the top right corner.
3. Select **"Save & Run All (Commit)"**.
4. Click **Save**.

## Step 4: Download Your Results
1. Wait for the version status to show **"Success"** (check the "Activity" tab of the notebook).
2. Click on the version number.
3. Go to the **Output** section.
4. Download your professional Excel file: `odoo_19_full_dataset_ai.xlsx`.

---

**Note on Hybrid Speed:**
The notebook automatically uses Groq for the first 14,000 items (taking ~15 mins) and then switches to the local Kaggle GPU to finish the rest without hitting daily limits.
