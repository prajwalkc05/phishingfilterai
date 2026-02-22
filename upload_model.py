from huggingface_hub import upload_folder

upload_folder(
    folder_path="final_model",
    repo_id="prajwalkc/phishing-bert",
    repo_type="model"
)

print("Model uploaded successfully")