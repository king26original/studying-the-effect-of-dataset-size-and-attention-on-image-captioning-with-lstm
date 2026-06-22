import torch
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from src.data.flickr8k.get_loader import load, transform

def evaluate(enc, dec, test_captions, itos, sample_fn, img_dir):
    """
    Evaluate model using METEOR and CIDEr metrics.
    
    Args:
        encoder: Trained encoder model
        decoder: Trained decoder model
        test_captions: Dictionary mapping image IDs to list of reference captions
        itos: Index to string vocabulary mapping
        sample_fn: Decoding function (e.g., decoder.sample or decoder.beam)
        img_dir: Directory containing test images
    
    Returns:
        Tuple of (METEOR score, CIDEr score)
    
    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc.eval()
    dec.eval()

    predictions={}
    with torch.no_grad():
        for image_id in test_captions.keys():
            image=load(image_id, img_dir)
            image=transform(image).unsqueeze(0).to(device)
            features=enc(image)
            output_ids=sample_fn(features)
            ans=""

            for id in output_ids:
                word=itos[id]
                if word in ['<start>', '<pad>', '<unk>']:
                    continue
                elif word=='<end>':
                    break
                ans+=word+' '
            ans=ans.strip() # removes trailig space
            predictions[image_id]=[ans]

    meteor=Meteor()
    meteor_score, _=meteor.compute_score(test_captions, predictions)

    cider=Cider()
    cider_score, _=cider.compute_score(test_captions, predictions)
    return meteor_score, cider_score