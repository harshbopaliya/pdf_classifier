import os
import shutil

def move_all_pdfs_to_root(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    count = 0

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                source_path = os.path.join(root, file)
                filename = os.path.basename(source_path)

                # Prevent overwriting by appending count if file exists
                while os.path.exists(os.path.join(target_dir, filename)):
                    name, ext = os.path.splitext(filename)
                    filename = f"{name}_{count}{ext}"
                    count += 1

                target_path = os.path.join(target_dir, filename)
                shutil.move(source_path, target_path)
                print(f"Moved: {source_path} -> {target_path}")

    print("\n✅ All PDFs moved.")

# Example usage:
source_directory = r"Policy"     # Source folder with nested structure
target_directory = r"."     # Destination where all PDFs will be collected

move_all_pdfs_to_root(source_directory, target_directory)
