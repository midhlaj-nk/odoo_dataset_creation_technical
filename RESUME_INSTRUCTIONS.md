# How to Resume Generation (If 12h Time Limit Reached)

Since the dataset generation takes ~24 hours and Kaggle limits sessions to 12 hours, you will need to run this in 2-3 sessions.

**Do not worry!** The script has built-in Checkpointing. You will not lose your work.

### Step 1: Check Session 1 Status
1. When your first 12-hour run finishes (or fails/timeouts), go to the notebook viewing page.
2. Click on the **Output** tab.
3. Look for a file named: `odoo_19_full_dataset_ai.xlsx.checkpoint.csv`.
   * *Note: This is the raw progress file. The final .xlsx might not be there yet if it timed out.*
4. **Download** this `.checkpoint.csv` file to your computer.

### Step 2: Prepare Session 2
1. Open your Kaggle Notebook again (Edit).
2. Click **"Add Input"** (or "Add Data") -> **"Upload"**.
3. Upload the `.checkpoint.csv` file you just downloaded.
   * Dataset Name: `odoo_checkpoint_1` (or anything you like).
   * Click **Create**.

### Step 3: Copy Checkpoint to Working Directory
You need to move the uploaded file (which is read-only) to the writable `/kaggle/working` directory so the script can resume from it.

Add this code block **before** you run the generation script:

```python
# Restore Checkpoint for Resume
import shutil
import os

# Adjust this path based on where Kaggle put your uploaded file.
# Usually it is under /kaggle/input/dataset-name/...
input_checkpoint = "/kaggle/input/odoo_checkpoint_1/odoo_19_full_dataset_ai.xlsx.checkpoint.csv"
working_checkpoint = "/kaggle/working/odoo_19_full_dataset_ai.xlsx.checkpoint.csv"

if os.path.exists(input_checkpoint):
    print(f"Restoring checkpoint from {input_checkpoint}...")
    shutil.copy(input_checkpoint, working_checkpoint)
    print("Checkpoint restored! Script will resume automatically.")
else:
    print("No checkpoint found in input. Starting fresh.")
```

### Step 4: Run Again
1. Click **"Save Version"** -> **"Save & Run All (Commit)"** again.
2. The script will see the file in `/kaggle/working`, print `Resuming from checkpoint...`, and skip everything you've already done.

Repeat this process until the script finishes completely and produces the final `.xlsx` file.
