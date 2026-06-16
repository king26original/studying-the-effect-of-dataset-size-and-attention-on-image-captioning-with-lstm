import spacy
from src.data.flickr8k.get_captions import captions, test_captions

spacy_eng=spacy.load("en_core_web_sm")

def tokenizer_eng(text):
    return [token.text.lower() for token in spacy_eng(text)]

data=[]

itos={0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
stoi={"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
idx=4

MAX_LEN=20

for key in captions.keys():
    for caption in captions[key]:
        tokens=tokenizer_eng(caption)
        
        for word in tokens:
            if word not in stoi:
                stoi[word]=idx
                itos[idx]=word
                idx+=1
                
        num=[stoi["<start>"]]
        num+=[stoi[word] for word in tokens]
        num.append(stoi["<end>"])
        
        if len(num)<=MAX_LEN:
            num=num+[stoi["<pad>"]]*(MAX_LEN-len(num))
        else:
            num=num[:MAX_LEN]
            num[-1]=stoi["<end>"]            
        data.append((key, num))

test_data=[]
for key in test_captions.keys():
    test_data.append((key, test_captions[key]))