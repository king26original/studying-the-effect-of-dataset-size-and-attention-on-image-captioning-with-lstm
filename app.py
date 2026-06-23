import streamlit as st
import torch
from PIL import Image
from src.model.enc_dec import encoder, decoder
from src.model.enc_dec_att import encoder_att, decoder_att
from inference import download_models, load_vocab, load_models, generate_caption

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Captioning",
    page_icon="🖼️",
    layout="centered",
)

# ── Load pipeline (runs once, cached for all users) ───────────────────────────
@st.cache_resource
def load_pipeline():
    """
    @st.cache_resource means Streamlit runs this function exactly once
    when the app first starts, then reuses the result for every user
    and every interaction. Without this, the model would reload on
    every button click, making the app unusably slow.
    """
    with st.spinner("Downloading model weights..."):
        encoder_attention_flickr30k_path, decoder_attention_flickr30k_path, encoder_no_attention_flickr30k_path, decoder_no_attention_flickr30k_path, encoder_no_attention_flickr8k_path, decoder_no_attention_flickr8k_path, vocab30k_path, vocab8k_path = download_models()

    itos_30k, stoi_30k = load_vocab(vocab30k_path)
    itos_8k, stoi_8k = load_vocab(vocab8k_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_30k_att, dec_30k_att = load_models(encoder_attention_flickr30k_path, decoder_attention_flickr30k_path, encoder_att, decoder_att, len(itos_30k), device)
    enc_30k_no_att, dec_30k_no_att = load_models(encoder_no_attention_flickr30k_path, decoder_no_attention_flickr30k_path,encoder, decoder, len(itos_30k), device)
    enc_8k_no_att, dec_8k_no_att = load_models(encoder_no_attention_flickr8k_path, decoder_no_attention_flickr8k_path,encoder, decoder, len(itos_8k), device)

    return enc_30k_att, dec_30k_att, enc_30k_no_att, dec_30k_no_att, enc_8k_no_att, dec_8k_no_att, itos_30k, itos_8k


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Image Captioning Demo")
st.write(
    "Upload an image and the model will generate a caption using a "
    "CNN-LSTM architecture with scaled dot-product attention."
)

enc_30k_att, dec_30k_att, enc_30k_no_att, dec_30k_no_att, enc_8k_no_att, dec_8k_no_att, itos_30k, itos_8k = load_pipeline()

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

beam_size = st.select_slider(
    "Beam size",
    options=[1, 3, 5, 7],
    value=3,
    help="1 = greedy (fastest). 3, 5, 7 = beam search (more careful, slightly slower).",
)

chosen_model= st.select_slider(
    "Chose the model",
    options=['trained on flickr30k with attention', 'trained on flickr30k without attention', 'trained on flickr8k without attention'],
    value='trained on flickr30k with attention'
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)
    
    with st.spinner("Generating caption..."):
        if chosen_model=='trained on flickr30k with attention':
            caption = generate_caption(image, enc_30k_att, dec_30k_att, itos_30k, beam_size=beam_size)
        elif chosen_model=='trained on flickr30k without attention':
            caption=generate_caption(image, enc_30k_no_att, dec_30k_no_att, itos_30k, beam_size=beam_size)
        elif chosen_model=='trained on flickr8k without attention':
            caption=generate_caption(image, enc_8k_no_att, dec_8k_no_att, itos_8k, beam_size=beam_size)

    st.success(f"**Caption:** {caption}")