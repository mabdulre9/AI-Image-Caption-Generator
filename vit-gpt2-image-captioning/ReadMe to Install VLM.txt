Use this code to install Vision Language Model from Colab used in this project


# Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Install huggingface_hub library
!pip install huggingface_hub

# Step 3: Download the model using snapshot_download
from huggingface_hub import snapshot_download


# Step 4: Define the model name and save path
model_name = "nlpconnect/vit-gpt2-image-captioning"
save_path = "/content/drive/MyDrive/vit-gpt2-image-captioning"  # Change if needed

# Step 5: Download the model
snapshot_download(repo_id=model_name, local_dir=save_path, local_dir_use_symlinks=False)


# The model is saved in your drive and you can download it from there and paste it in this folder
