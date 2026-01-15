# How to Generate the Odoo Dataset using Google Colab

Google Colab is a great alternative to Kaggle for running the AI generation task on free Cloud GPUs.

## Step 1: Prepare your Drive
1. Go to [drive.google.com](https://drive.google.com).
2. Create a folder called `odoo_dataset_creation`.
3. Upload the **`generate_odoo_dataset.py`** script to that folder.

## Step 2: Open and Run
1. Go to [colab.research.google.com](https://colab.research.google.com/).
2. Click **File** > **Upload Notebook** and select **`Run_on_Google_Colab.ipynb`**.
3. **Important**: Go to **Runtime** > **Change runtime type** and ensure **T4 GPU** is selected.

## Step 3: Configure & Start
1. Run the first cell to install dependencies.
2. In the configuration cell, you can optionally add your Groq API key for faster initial processing.
3. Run all cells (`Ctrl + F9`).
4. Follow the prompt to "Mount Google Drive" so the script can save your result.

## Step 4: Completion
The final Excel file will be saved directly to your `odoo_dataset_creation` folder on Google Drive.
