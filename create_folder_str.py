import os

def create_project_folders(base_path='.'):
    folders = [
        'data',
        'logs',
        'results',
        'results/colored_img',
        'saved_models'
    ]

    for folder in folders:
        path = os.path.join(base_path, folder)
        os.makedirs(path, exist_ok=True)
        print(f"Created folder: {path}")

if __name__ == "__main__":
    create_project_folders()