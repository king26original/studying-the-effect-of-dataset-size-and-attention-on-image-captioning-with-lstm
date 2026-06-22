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
  img_path=os.path.join(img_dir, img_name)
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

def fn(batch):
  img=[]
  captions=[]
  for image_name, caption in batch:
    img.append(load(image_name, IMG_DIR, transform=transform))
    captions.append(torch.tensor(caption, dtype=torch.long))
    
  captions=pad_sequence(captions, batch_first=True, padding_value=0)  
  return torch.stack(img), captions

def create_train_loader(train_df):
    data_custom=cust_dataset(train_df, feature='image_name', label='numerical_caption')
    train_loader=DataLoader(data_custom, batch_size=32, shuffle=True, collate_fn=fn)
    return train_loader