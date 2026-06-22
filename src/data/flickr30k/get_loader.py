from torchvision import transforms
from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# update this according the location
IMG_DIR = "/path/to/flickr30k/images"

#for transforming into standard size and converting from pil into tensor
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

#for getting the image from the image name
def load(img_name, img_dir, transform=transform):
  """
    Load and preprocess an image.
    
    Args:
        img_name: Image filename (e.g., "12345.jpg")
        img_dir: Directory containing images
        transform: Optional torchvision transforms to apply
    
    Returns:
        Preprocessed image tensor
    
    Raises:
        FileNotFoundError: If image file doesn't exist
  """

  img_path=os.path.join(img_dir, img_name)
  if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found: {img_path}")

  img=Image.open(img_path).convert("RGB")
  if transform is not None:
    img=transform(img)
  return img

class cust_dataset(Dataset):
    def __init__(self, df, feature, label):
        self.image=df[feature].tolist()
        self.caption=df[label].tolist()
        
    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        return self.image[idx], self.caption[idx]

def fn(batch, img_dir=IMG_DIR):
  """
    Collate function for DataLoader.
    
    Loads images and pads captions to same length.
    
    Args:
        batch: List of (image_name, caption) tuples
        img_dir: Image directory path
    
    Returns:
        Tuple of (images, padded_captions) tensors
  """

  img=[]
  captions=[]
  for image_name, caption in batch:
    img.append(load(image_name, img_dir, transform=transform))
    captions.append(torch.tensor(caption, dtype=torch.long))
    
  captions=pad_sequence(captions, batch_first=True, padding_value=0)  
  return torch.stack(img), captions

def create_train_loader(train_df, batch_size,num_workers=0, shuffle=True):
    """
    Create a DataLoader for training.
    
    Args:
        train_df: Training DataFrame
        batch_size: Batch size (default: 32)
        shuffle: Whether to shuffle data (default: True)
        num_workers: Number of worker processes (default: 0)
    
    Returns:
        Configured DataLoader
    """

    data_custom=cust_dataset(train_df, feature='image_name', label='numerical_caption')
    train_loader=DataLoader(data_custom, batch_size, shuffle, collate_fn=fn, num_workers=num_workers,pin_memory=True)
    return train_loader