import pandas as pd
import spacy
from collections import Counter

spacy_eng=spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer", "textcat"])

def tokenize_text(text):
    return [tok.text.lower() for tok in spacy_eng(str(text))]

def build_vocab(tokenized_series, freq_threshold=5):
    itos={0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
    stoi={"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
    idx=4
    
    frequencies=Counter()
    for tokens in tokenized_series:
        frequencies.update(tokens)
        
    for word, count in frequencies.items():
        if count>=freq_threshold:
            stoi[word]=idx
            itos[idx]=word
            idx+=1
            
    return stoi, itos

def numericalize_tokens(tokens, stoi):
    return [stoi["<start>"]]+[stoi.get(word, stoi["<unk>"]) for word in tokens]+[stoi["<end>"]]

def create_tokens(train_df):
    train_df['tokens']=train_df[' comment'].apply(tokenize_text)
    stoi, itos=build_vocab(train_df['tokens'], freq_threshold=5)
    train_df['numerical_caption']=train_df['tokens'].apply(lambda x: numericalize_tokens(x, stoi))