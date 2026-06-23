import torch
import pickle
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
from src.model.enc_dec_att import encoder, decoder
from src.data.flickr8k.get_loader import transform

ENC_DIM=256
EMBED_SIZE=256
HIDDEN_SIZE=512
ATTENTION_DIM=512

# ── HuggingFace repo ─────────────────────────────────────────────────────────
# Replace with your actual HuggingFace username/repo-name
HF_REPO_ID = "king11010/image-captioning"

def download_models():
    """
    Download model weights and vocab from HuggingFace Hub.
    hf_hub_download caches files locally after the first download,
    so this is fast on subsequent calls.

    Returns:
        Tuple of (encoder_path, decoder_path, vocab_path) as strings
    """
    encoder_attention_flickr30k_path = hf_hub_download(repo_id=HF_REPO_ID, filename="enc-lstm-attention-flckr30k.pth")
    decoder_attention_flickr30k_path = hf_hub_download(repo_id=HF_REPO_ID, filename="dec-lstm-attention-flckr30k.pth")
    encoder_no_attention_flickr30k_path = hf_hub_download(repo_id=HF_REPO_ID, filename="enc-no-lstm-attention-flckr30k.pth")
    decoder_no_attention_flickr30k_path = hf_hub_download(repo_id=HF_REPO_ID, filename="dec-no-lstm-attention-flckr30k.pth")
    encoder_no_attention_flickr8k_path = hf_hub_download(repo_id=HF_REPO_ID, filename="enc-no-lstm-attention-flckr8k.pth")
    decoder_no_attention_flickr8k_path = hf_hub_download(repo_id=HF_REPO_ID, filename="dec-no-lstm-attention-flckr8k.pth")
    vocab30k_path   = hf_hub_download(repo_id=HF_REPO_ID, filename="vocab30k_attention.pkl")
    vocab8k_path   = hf_hub_download(repo_id=HF_REPO_ID, filename="vocab8k.pkl")
    return encoder_attention_flickr30k_path, decoder_attention_flickr30k_path, encoder_no_attention_flickr30k_path, decoder_no_attention_flickr30k_path, encoder_no_attention_flickr8k_path, decoder_no_attention_flickr8k_path, vocab30k_path, vocab8k_path


def load_vocab(vocab_path):
    """
    Load the itos and stoi vocabulary dicts from a pickle file.

    Args:
        vocab_path: Path to vocab.pkl

    Returns:
        Tuple of (itos dict, stoi dict)
    """
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    return vocab["itos"], vocab["stoi"]


def load_models(encoder_path, decoder_path, vocab_size, device):
    """
    Reconstruct encoder and decoder with the same architecture used
    during training, then load the saved weights.

    Args:
        encoder_path: Path to encoder.pth
        decoder_path: Path to decoder.pth
        vocab_size:   Number of words in vocabulary (must match training)
        device:       torch.device to load models onto

    Returns:
        Tuple of (encoder, decoder), both in eval mode
    """
    # Build encoder with same enc_dim as training
    enc = encoder(ENC_DIM)
    enc.load_state_dict(torch.load(encoder_path, map_location=device))
    enc.to(device)
    enc.eval()  # switches BatchNorm to use running stats, not batch stats

    # Build decoder with same hyperparams as training
    dec = decoder(
        embed_size    = EMBED_SIZE,
        hidden_size   = HIDDEN_SIZE,
        vocab_size    = vocab_size,
        encoder_dim   = ENC_DIM,
        attention_dim = ATTENTION_DIM,
    )
    dec.load_state_dict(torch.load(decoder_path, map_location=device))
    dec.to(device)
    dec.eval()

    return enc, dec


def generate_caption(image, enc, dec, itos, beam_size=3, max_len=20):
    """
    Run inference on a single PIL image and return a caption string.

    Args:
        image:     PIL Image object
        enc:       Loaded encoder model
        dec:       Loaded decoder model
        itos:      Index-to-string vocab dict
        beam_size: 1 = greedy decoding, 3/5/7 = beam search
        max_len:   Maximum number of tokens to generate

    Returns:
        Caption as a plain string
    """
    device = next(enc.parameters()).device

    # Preprocess: PIL Image → (1, 3, 224, 224) tensor
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = enc(image_tensor)  # (1, 49, 256)

        if beam_size == 1:
            token_ids = dec.sample(features, max_len=max_len)
        else:
            token_ids = dec.beam(features, k=beam_size, max_len=max_len)

    # Convert token IDs back to words, skipping special tokens
    words = []
    for token_id in token_ids:
        word = itos[token_id]
        if word in ("<start>", "<pad>", "<unk>"):
            continue
        if word == "<end>":
            break
        words.append(word)

    return " ".join(words)