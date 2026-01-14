# How to Generate the Odoo Dataset using Kaggle (Background Run)

This guide helps you run the 24-hour job in the background using Kaggle's free GPU instances.

## Step 1: Push to GitHub
1. Create a **Public** GitHub repository (e.g., `odoo_dataset_creation`).
2. Push your `odoo_dataset_creation` folder to it.
   - It should contain `generate_odoo_dataset.py`.
   - *You do NOT need to push the odoo-19.0 folder.*

## Step 2: Create a Kaggle Notebook
1. Go to [kaggle.com](https://www.kaggle.com/) and Log in.
2. Click **Create** > **New Notebook**.
3. **Settings (Important)**:
   - Expand the "Settings" menu on the right.
   - **Internet**: Switch to **On** (Requires phone verification if new account).
   - **Accelerator**: Select **GPU T4 x2** (or GPU P100).
   - ⚠️ **IMPORTANT**: Do **NOT** select "TPU". Ollama does not run on TPUs. If you select TPU, the script will fail.

## Step 3: Setup the Notebook
1. In the first cell of the Kaggle Notebook, clone your GitHub repo:
   ```python
   !git clone https://github.com/YOUR_USERNAME/odoo_dataset_creation.git
   ```
2. In the separate cells, copy the commands to install Ollama and run the script.
   
   **Cell 2 (Install):**
   ```python
   !curl -fsSL https://ollama.com/install.sh | sh
   !pip install pandas openpyxl requests tqdm openai
   ```
   
   **Cell 3 (Start Server):**
   ```python
   import subprocess
   import time
   subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
   time.sleep(10)
   !ollama pull llama3
   ```
   
   **Cell 4 (Get Odoo):**
   ```python
   !git clone --depth 1 --branch 19.0 https://github.com/odoo/odoo.git odoo-19.0
   ```
   
   **Cell 5 (Run):**
   ```python
   # Paths
   script = "./odoo_dataset_creation/generate_odoo_dataset.py"
   odoo = "./odoo-19.0"
   output = "odoo_19_full_dataset_ai.xlsx"
   
   # Run
   !python3 $script $odoo $output --ollama-model llama3
   ```

## Step 4: Run in Background (Vacation Mode)
1. Do **NOT** just press the usage "Play" button.
2. Click the **"Save Version"** button in the top right corner.
3. Select **"Save & Run All (Commit)"**.
4. Click **Save**.

**What happens now?**
- Kaggle spins up a separate server in the background.
- It will run your script for up to **12 hours**, even if you close your laptop.
- When it finishes (or times out), it saves the output files (`xlsx` and `checkpoint.csv`).
- You can check the status by clicking on the notebook name -> "Activity" or "Versions".
- When "Success", go to the "Output" tab of that version and download your Excel file.
