import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import nibabel as nib 


SAVE_DIR = "/home/woody/iwi5/iwi5207h/case_study/data"
os.makedirs(SAVE_DIR, exist_ok=True)

def load_image(file_path):
    try:
        img = nib.load(file_path)
        return img.get_fdata()
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

def display_slices(image, seg=None, title="Image", save_name=None):
    if image is None:
        print(f"Skipping {title} due to loading error.")
        return

    mid_x, mid_y, mid_z = np.array(image.shape) // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image[mid_x, :, :], cmap='gray')  # Sagittal
    axes[1].imshow(image[:, mid_y, :], cmap='gray')  # Coronal
    axes[2].imshow(image[:, :, mid_z], cmap='gray')  # Axial

    if seg is not None and seg.shape == image.shape:
        for i, ax in enumerate(axes):
            overlay = seg[mid_x, :, :] if i == 0 else seg[:, mid_y, :] if i == 1 else seg[:, :, mid_z]
            ax.imshow(overlay, cmap='Reds', alpha=0.5)  # Overlay segmentation

    for ax in axes:
        ax.axis('off')

    plt.suptitle(title)

    if save_name:
        save_path = os.path.join(SAVE_DIR, f"{save_name}.png")
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.close() 

def load_and_display_images(img_dir, seg_dir, img_pattern='*.nii', seg_pattern='*.nii', max_pairs=5):
    
    img_paths = sorted(glob.glob(os.path.join(img_dir, img_pattern)))[:max_pairs]
    seg_paths = sorted(glob.glob(os.path.join(seg_dir, seg_pattern)))[:max_pairs]

    print(f"Found {len(img_paths)} images: {img_paths}")
    print(f"Found {len(seg_paths)} segmentations: {seg_paths}")

    if len(img_paths) != len(seg_paths):
        print("Mismatch in number of images and segmentations!")

    for img_path, seg_path in zip(img_paths, seg_paths):
        image_data = load_image(img_path)
        seg_data = load_image(seg_path)

        img_name = os.path.basename(img_path).replace('.nii', '')
        seg_name = os.path.basename(seg_path).replace('.nii', '')

        # Save the original image
        display_slices(image_data, title=f"Image: {img_name}", save_name=f"image_{img_name}")

        # Save segmentation image
        display_slices(seg_data, title=f"Segmentation: {seg_name}", save_name=f"seg_{seg_name}")

        # Save overlay image
        if image_data is not None and seg_data is not None:
            display_slices(image_data, seg_data, 
                                 title=f"Overlay: {img_name} & {seg_name}", 
                                 save_name=f"overlay_{img_name}_{seg_name}")

if __name__ == "__main__":
    img_dir = "/home/woody/iwi5/iwi5207h/case_study/data/extracted/img"
    seg_dir = "/home/woody/iwi5/iwi5207h/case_study/data/extracted/seg"

    load_and_display_images(img_dir, seg_dir)
