from src.data.flickr8k.download_flickr8k import download

def get_captions(train_file, test_file, caption_file):
  with open(train_file) as f:
    images=[line.strip() for line in f]

  with open(test_file) as f:
    test_images=[line.strip() for line in f]

  images=set(images)
  test_images=set(test_images)

  captions={}
  test_captions={}

  with open(caption_file) as f:
    for line in f:
      image_id,caption=line.strip().split('\t')
      image_id=image_id.split('#')[0]

      if image_id in images:
        captions.setdefault(image_id, []).append(caption.lower())
      elif image_id in test_images:
        test_captions.setdefault(image_id, []).append(caption.lower())
  
  return captions, test_captions