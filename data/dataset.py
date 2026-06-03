import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CarvanaDataset(Dataset):
    """
    Custom Dataset class for loading Carvana image and mask data.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (str): Directory with all the images.
            mask_dir (str): Directory with all the masks.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        # List all image files in the image directory
        self.images = os.listdir(image_dir)

    def __len__(self):
        # Return the total number of image files
        return len(self.images)

    def __getitem__(self, index):
        # Get the name of the image file at the given index
        img_path = os.path.join(self.image_dir, self.images[index])
        # Construct the path to the corresponding mask file
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif")) # Corrected mask_dir

        # Open and convert the image to RGB format
        image = np.array(Image.open(img_path).convert("RGB"))
        # Open and convert the mask to grayscale (L) format and cast to float32
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # Normalize the mask values to be either 0.0 or 1.0
        mask[mask == 255.0] = 1.0

        # Apply transformations if specified
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        # Return the processed image and mask
        return image, mask