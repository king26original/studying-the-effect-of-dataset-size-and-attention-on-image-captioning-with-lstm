import numpy as np
import torch
from torch.nn.modules import loss
import torch.nn as nn
import logging
from datetime import datetime
from pathlib import Path

def setup_logger(name,log_dir,level,):
    """
    Set up logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
    
    Returns:
        Configured logger
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger=logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers =[]
    
    # File handler (detailed)
    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    fh=logging.FileHandler(f"{log_dir}/{name}_{timestamp}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',datefmt='%Y-%m-%d %H:%M:%S'))
    
    # Console handler (summary)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def train(encoder, decoder, train_loader, itos, stoi, att=False):
    """
    Train the image captioning model.
    
    Args:
        encoder_class: Encoder class (not instance)
        decoder_class: Decoder class (not instance)
        train_loader: DataLoader for training data
        itos: Index to string vocabulary mapping
        stoi: String to index vocabulary mapping
        att: Whether to use attention mechanism (default: False)
        config: Optional configuration dictionary with:
            - enc_dim: Encoder dimension (default: 256)
            - embed_size: Embedding size (default: 256)
            - hidden_size: LSTM hidden size (default: 512)
            - attention_dim: Attention dimension (default: 512)
            - num_epochs: Number of epochs (default: 80)
            - learning_rate: Learning rate (default: 0.001)
    
    Returns:
        Tuple containing:
            - losses: List of epoch losses
            - encoder: Trained encoder model
            - decoder: Trained decoder model
    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger=setup_logger("training")
    config={}

    logger.info(f"Using device: {device}")
    
    # Hyperparameters
    ENC_DIM=config.get('enc_dim', 256)
    EMBED_SIZE=config.get('embed_size', 256)
    HIDDEN_SIZE=config.get('hidden_size', 512)
    ATTENTION_DIM=config.get('attention_dim', 512)
    NUM_EPOCHS=config.get('num_epochs', 80)
    LEARNING_RATE=config.get('learning_rate', 0.001)
    
    logger.info(f"Configuration: {config}")
    logger.info(f"Vocabulary size: {len(itos)}")
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Using attention: {att}")

    
    enc=encoder(ENC_DIM).to(device)
    dec=decoder(EMBED_SIZE,HIDDEN_SIZE,len(itos)).to(device) if att is False else decoder(embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, vocab_size=len(itos), encoder_dim=ENC_DIM, attention_dim=ATTENTION_DIM).to(device)

    loss_fn=nn.CrossEntropyLoss(ignore_index=stoi["<pad>"])
    optimizer=torch.optim.Adam(list(enc.parameters())+list(dec.parameters()), lr=LEARNING_RATE)
    
    enc.train()
    dec.train()

    scaler=torch.cuda.amp.GradScaler()

    losses=[]
    best_loss=float('inf')
    
    logger.info("Starting training...")

    for epoch in range(NUM_EPOCHS):
        loss_epoch=0
        for idx, (images,captions) in enumerate(train_loader):
            images=images.to(device)
            captions=captions.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                features=enc(images)
                output=dec(features,captions)
                if att:
                    targets = captions[:, 1:]
                else:
                    targets = captions

                loss=loss_fn(output.view(-1, len(itos)), targets.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if idx%100==0:
                logger.debug(
                    f"Epoch {epoch+1}/{NUM_EPOCHS} | "
                    f"Batch {idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

            loss_epoch=loss.item()+loss_epoch
        loss_epoch=loss_epoch/len(train_loader)
        losses.append(loss_epoch)

        logger.info(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Avg Loss: {loss_epoch:.4f} | "
            f"Best: {min(losses):.4f}"
        )

        if loss_epoch < best_loss:
            best_loss = loss_epoch
            logger.info(f"New best loss! Saving checkpoint...")
    
    return losses, enc, dec