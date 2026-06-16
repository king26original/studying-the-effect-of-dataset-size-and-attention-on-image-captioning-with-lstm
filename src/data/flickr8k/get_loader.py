from torchvision import transforms
from PIL import Image
import os
import torch
from src.data.flickr8k.tokenize import data

#for transforming into standard size and converting from pil into tensor
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

#for getting the image from the image name
def load(img_name, img_dir, transform=None):
  img_path=os.path.join(img_dir, img_name)
  img=Image.open(img_path).convert("RGB")
  if transform is not None:
    img=transform(img)
  return img

def fn(batch):
  img=[]
  captions=[]
  for image_name, caption in batch:
    img.append(load(image_name, transform=transform))
    captions.append(torch.tensor(caption, dtype=torch.long))

  captions=torch.stack(captions)
  return torch.stack(img), captions

from torch.utils.data import Dataset, DataLoader

train_loader=DataLoader(data, batch_size=32, shuffle=True, collate_fn=fn)