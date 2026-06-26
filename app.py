# import streamlit as st
# import torch
# from PIL import Image
# from src.model.enc_dec import encoder, decoder
# from src.model.enc_dec_att import encoder_att, decoder_att
# from inference import download_models, load_vocab, load_models, generate_caption

# # ── Page config ───────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Image Captioning",
#     page_icon="🖼️",
#     layout="centered",
# )

# # ── Load pipeline (runs once, cached for all users) ───────────────────────────
# @st.cache_resource
# def load_pipeline():
#     """
#     @st.cache_resource means Streamlit runs this function exactly once
#     when the app first starts, then reuses the result for every user
#     and every interaction. Without this, the model would reload on
#     every button click, making the app unusably slow.
#     """
#     with st.spinner("Downloading model weights..."):
#         encoder_attention_flickr30k_path, decoder_attention_flickr30k_path, encoder_no_attention_flickr30k_path, decoder_no_attention_flickr30k_path, encoder_no_attention_flickr8k_path, decoder_no_attention_flickr8k_path, vocab30k_path, vocab8k_path = download_models()

#     itos_30k, stoi_30k = load_vocab(vocab30k_path)
#     itos_8k, stoi_8k = load_vocab(vocab8k_path)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     enc_30k_att, dec_30k_att = load_models(encoder_attention_flickr30k_path, decoder_attention_flickr30k_path, encoder_att, decoder_att, len(itos_30k), device)
#     enc_30k_no_att, dec_30k_no_att = load_models(encoder_no_attention_flickr30k_path, decoder_no_attention_flickr30k_path,encoder, decoder, len(itos_30k), device, att=False)
#     enc_8k_no_att, dec_8k_no_att = load_models(encoder_no_attention_flickr8k_path, decoder_no_attention_flickr8k_path,encoder, decoder, len(itos_8k), device, att=False)

#     return enc_30k_att, dec_30k_att, enc_30k_no_att, dec_30k_no_att, enc_8k_no_att, dec_8k_no_att, itos_30k, itos_8k


# # ── UI ────────────────────────────────────────────────────────────────────────
# st.title("Image Captioning Demo")
# st.write(
#     "Upload an image and the model will generate a caption using a "
#     "CNN-LSTM architecture with scaled dot-product attention."
# )

# enc_30k_att, dec_30k_att, enc_30k_no_att, dec_30k_no_att, enc_8k_no_att, dec_8k_no_att, itos_30k, itos_8k = load_pipeline()

# uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# beam_size = st.select_slider(
#     "Beam size",
#     options=[1, 3, 5, 7],
#     value=3,
#     help="1 = greedy (fastest). 3, 5, 7 = beam search (more careful, slightly slower).",
# )

# chosen_model= st.select_slider(
#     "Chose the model",
#     options=['trained on flickr30k with attention', 'trained on flickr30k without attention', 'trained on flickr8k without attention'],
#     value='trained on flickr30k with attention'
# )

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, use_column_width=True)
    
#     with st.spinner("Generating caption..."):
#         caption=None
#         if chosen_model=='trained on flickr30k with attention':
#             caption = generate_caption(image, enc_30k_att, dec_30k_att, itos_30k, beam_size=beam_size)
#         elif chosen_model=='trained on flickr30k without attention':
#             caption=generate_caption(image, enc_30k_no_att, dec_30k_no_att, itos_30k, beam_size=beam_size)
#         elif chosen_model=='trained on flickr8k without attention':
#             caption=generate_caption(image, enc_8k_no_att, dec_8k_no_att, itos_8k, beam_size=beam_size)

#     st.success(f"**Caption:** {caption}")

import os
import time
from datetime import datetime

import streamlit as st
import torch
import pandas as pd
from PIL import Image
from src.model.enc_dec import encoder, decoder
from src.model.enc_dec_att import encoder_att, decoder_att
from inference import download_models, load_vocab, load_models, generate_caption

from db import init_db, get_session, Image as DBImage, Caption
from caching import hash_image_bytes, find_existing_image, find_cached_caption

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Captioning",
    page_icon="🖼️",
    layout="centered",
)

init_db()
UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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
    enc_30k_no_att, dec_30k_no_att = load_models(encoder_no_attention_flickr30k_path, decoder_no_attention_flickr30k_path,encoder, decoder, len(itos_30k), device, att=False)
    enc_8k_no_att, dec_8k_no_att = load_models(encoder_no_attention_flickr8k_path, decoder_no_attention_flickr8k_path,encoder, decoder, len(itos_8k), device, att=False)

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
    image_bytes = uploaded_file.getvalue()
    image_hash = hash_image_bytes(image_bytes)
    decoding_strategy = "greedy" if beam_size == 1 else "beam"
    beam_width = None if beam_size == 1 else beam_size

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    session = get_session()
    image_row = find_existing_image(session, image_hash)
    if image_row is None:
        file_path = os.path.join(UPLOAD_DIR, f"{image_hash}.jpg")
        image.save(file_path)
        image_row = DBImage(sha256_hash=image_hash, original_filename=uploaded_file.name, file_path=file_path)
        session.add(image_row)
        session.commit()

    cached = find_cached_caption(session, image_row.id, chosen_model, decoding_strategy, beam_width)

    if cached is not None:
        caption, latency_ms = cached.caption_text, cached.latency_ms
    else:
        with st.spinner("Generating caption..."):
            start = time.perf_counter()
            caption=None
            if chosen_model=='trained on flickr30k with attention':
                caption = generate_caption(image, enc_30k_att, dec_30k_att, itos_30k, beam_size=beam_size)
            elif chosen_model=='trained on flickr30k without attention':
                caption=generate_caption(image, enc_30k_no_att, dec_30k_no_att, itos_30k, beam_size=beam_size)
            elif chosen_model=='trained on flickr8k without attention':
                caption=generate_caption(image, enc_8k_no_att, dec_8k_no_att, itos_8k, beam_size=beam_size)
            latency_ms = (time.perf_counter() - start) * 1000

        session.add(Caption(
            image_id=image_row.id, model_variant=chosen_model,
            decoding_strategy=decoding_strategy, beam_width=beam_width,
            caption_text=caption, latency_ms=latency_ms,
        ))
        session.commit()

    session.close()

    st.success(f"**Caption:** {caption}")
    st.caption(f"Latency: {latency_ms:.1f} ms")

# ── Caption history (searchable + CSV export) ─────────────────────────────────
st.header("Caption history")

session = get_session()
rows = (
    session.query(Caption, DBImage)
    .join(DBImage, Caption.image_id == DBImage.id)
    .order_by(Caption.created_at.desc())
    .all()
)
history = [
    {
        "filename": img.original_filename,
        "model": cap.model_variant,
        "decoding_strategy": cap.decoding_strategy,
        "beam_width": cap.beam_width,
        "caption": cap.caption_text,
        "latency_ms": round(cap.latency_ms, 1),
        "created_at": cap.created_at,
    }
    for cap, img in rows
]
session.close()

df = pd.DataFrame(history)
if not df.empty:
    search_term = st.text_input("Search captions (filename or caption text)")
    if search_term:
        df = df[
            df["filename"].str.contains(search_term, case=False, na=False)
            | df["caption"].str.contains(search_term, case=False, na=False)
        ]
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "Export to CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"captions_{datetime.now():%Y%m%d_%H%M%S}.csv",
        mime="text/csv",
    )