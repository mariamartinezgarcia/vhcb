
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CelebA
from torchvision import transforms
from PIL import Image
import os
import pandas as pd


# Function to load CelebA dataset with specific attributes
def get_celeba_dataloader(root_dir="./data", batch_size=64, selected_attributes=None, image_size=128, num_workers=4, split='train'):
    """
    Creates dataloader for the CelebA dataset.

    Parameters:
        root_dir (str): Directory where CelebA dataset will be downloaded/stored.
        batch_size (int): Batch size for the dataloaders.
        selected_attributes (list): List of attribute names to extract.
        image_size (int): Image resizing size.
        num_workers (int): Number of workers for data loading.
        split (string): data split we want ('train', 'test', or 'valid')

    Returns:
        dict: A dictionary containing 'train', 'val', and 'test' dataloaders.
    """
    
    # Define transforms for the images
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
    ])

    # Load the full CelebA dataset
    dataset = CelebA(root=root_dir, split=split, download=True, transform=transform)

    # Get attribute names
    attr_names = dataset.attr_names[:40]

    # Select specific attributes
    if selected_attributes:
        indices = [attr_names.index(attr) for attr in selected_attributes]
    else:
        indices = list(range(len(attr_names)))  # Use all attributes if none are selected

    # Get image indices for each split
    # Create subsets for train, validation, and test

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    return dataloader, attr_names, indices


class CelebASubset(Dataset):
    def __init__(self, attribute_file, image_dir, selected_attributes, transform=None, positive=True):
        """
        CelebA dataset subset loader based on attributes.

        Args:
            attribute_file (str): Path to 'list_attr_celeba.txt'.
            image_dir (str): Path to the CelebA image directory.
            selected_attributes (list): List of attribute names to filter by.
            transform (torchvision.transforms): Transformations for images.
            positive (bool): If True, selects images with the attributes;
                             If False, selects images without the attributes.
        """
        self.image_dir = image_dir
        self.transform = transform

        # Load attributes
        df = pd.read_csv(attribute_file, delim_whitespace=True, header=1)
        df = df.replace(-1, 0)  # Convert -1/1 to 0/1 for easier filtering

        # Filter dataset based on attributes
        mask = df[selected_attributes].all(axis=1) if positive else ~df[selected_attributes].any(axis=1)
        self.filtered_df = df[mask]  # Subset the DataFrame

        # Store image filenames and labels
        self.image_filenames = self.filtered_df.index.tolist()
        self.labels = self.filtered_df[selected_attributes].values  # Labels for selected attributes

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGB")  # Load image in RGB format
        label = self.labels[idx]  # Get corresponding label

        if self.transform:
            image = self.transform(image)  # Apply transformations

        return image, label