import os
import shutil
import tarfile
import zipfile

# Input and output directories
input_dir = "arxiv_source_files"
output_dir = "arxiv_sources"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to check if a directory contains a .tex file
def contains_tex_file(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".tex"):
                return True
    return False

# Counters for directories handled and processed
total_directories_handled = 0
total_directories_processed = 0

# Process each subdirectory in the input directory
for subdir in os.listdir(input_dir):
    subdir_path = os.path.join(input_dir, subdir)

    # Skip if not a directory
    if not os.path.isdir(subdir_path):
        continue

    total_directories_handled += 1

    # Process compressed files in the subdirectory
    for file in os.listdir(subdir_path):
        file_path = os.path.join(subdir_path, file)

        # Ensure processing only compressed files (tar, zip, etc.)
        try:
            # Output path for decompressed content
            output_subdir = os.path.join(output_dir, os.path.splitext(file)[0])

            if tarfile.is_tarfile(file_path):
                # Handle tar files
                with tarfile.open(file_path) as tar:
                    tar.extractall(output_subdir)

            elif zipfile.is_zipfile(file_path):
                # Handle zip files
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(output_subdir)
            else:
                print(f"{file_path} is not decompressible.")
                continue

            # Check for .tex files in the decompressed directory
            if not contains_tex_file(output_subdir):
                print(f"No .tex file found. Deleting directory {output_subdir}.")
                shutil.rmtree(output_subdir)
            else:
                total_directories_processed += 1

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# Print summary
print(f"Total directories handled: {total_directories_handled}")
print(f"Total directories processed: {total_directories_processed}")
