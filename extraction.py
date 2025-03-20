import os
import zipfile
import gzip
import shutil

zip_file_path = '/home/woody/iwi5/iwi5207h/case_study/data/drive-download-20230906T142405Z-001.zip'

base_extract_path = '/home/woody/iwi5/iwi5207h/case_study/data/extracted'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(base_extract_path)

img_folder = os.path.join(base_extract_path, 'img')
seg_folder = os.path.join(base_extract_path, 'seg')

def extract_and_remove_gz(folder):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.gz'):
                gz_path = os.path.join(root, file)
                extracted_path = os.path.join(root, file[:-3]) 
                
                with gzip.open(gz_path, 'rb') as f_in, open(extracted_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                
                os.remove(gz_path)
                print(f"Extracted and removed: {gz_path}")
                
extract_and_remove_gz(img_folder)
extract_and_remove_gz(seg_folder)

print("All images are extracted and .gz files are removed.")
