import numpy as np
import torch
from torch.nn.modules import loss
import torch.nn as nn

def train(encoder, decoder, train_loader, itos, stoi, att=False):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ENC_DIM=256
    EMBED_SIZE=256
    HIDDEN_SIZE=512
    NUM_LAYERS=1
    if att is True:
        ATTENTION_DIM=512
    
    enc=encoder(ENC_DIM).to(device)
    dec=decoder(EMBED_SIZE,HIDDEN_SIZE,len(itos)).to(device) if att is False else decoder(embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, vocab_size=len(itos), encoder_dim=ENC_DIM, attention_dim=ATTENTION_DIM)

    loss_fn=nn.CrossEntropyLoss(ignore_index=stoi["<pad>"])
    optimizer=torch.optim.Adam(list(enc.parameters())+list(dec.parameters()), lr=0.001)
    
    enc.train()
    dec.train()

    NUM_EPOCHS=80

    scaler=torch.cuda.amp.GradScaler()

    losses=[]

    for epoch in range(NUM_EPOCHS):
        loss_epoch=0
        for idx, (images,captions) in enumerate(train_loader):
            images=images.to(device)
            captions=captions.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                features=enc(images)
                output=dec(features,captions)
                loss=loss_fn(output.view(-1, len(itos)), captions.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if idx%100==0:
                print(f"epoch{epoch}, idx={idx}, loss: {loss.item()}")

            loss_epoch=loss.item()+loss_epoch
        loss_epoch=loss_epoch/len(train_loader)
        losses.append(loss_epoch)
    
    return losses, enc, dec