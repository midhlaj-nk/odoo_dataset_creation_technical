# How to Generate the Odoo Dataset using Google Colab

Since generating the dataset with separate descriptions for 35,000+ items takes a long time on a laptop (approx 24h), we recommend using Google Colab.

This update pulls the Odoo source code directly from GitHub, so you don't need to upload the large `odoo-19.0` folder manually.

## Prerequisites
- A Google Account.
- A tiny amount of space on Google Drive (just for the script and the result).

## Step 1: Upload ONE file
1. Go to [drive.google.com](https://drive.google.com).
2. Create a folder called `odoo_dataset_creation`.
3. Upload ONLY **`generate_odoo_dataset.py`** to that folder.

## Step 2: Open Google Colab
1. Go to [colab.research.google.com](https://colab.research.google.com/).
2. Upload the `Run_on_Google_Colab.ipynb` file found in this directory.

## Step 3: Configure GPU
1. **Runtime > Change runtime type**.
2. Select **T4 GPU**.
3. Save.

## Step 4: Run
1. Run all cells in order.
   - It will install tools.
   - It will **Download Odoo 19.0** automatically.
   - It will connect to your Drive to find the script.
   - it will run the AI generation.
   
## Step 5: Done
- The final Excel file will be saved to your `odoo_dataset_creation` folder on Google Drive.
