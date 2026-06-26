"""
we hash the raw bytes of every uploaded image with SHA-256. if two uploads
produce the same hash, they are the same image so instead
of rerunning an expensive model forward pass, we just look up whatever
caption we already generated for that exact (image, decoding config)
combination
"""

import hashlib
from typing import Optional

from db import Image, Caption


def hash_image_bytes(image_bytes: bytes) -> str:
    """return the SHA-256 hex digest of the raw image bytes"""
    return hashlib.sha256(image_bytes).hexdigest()


def find_existing_image(session, image_hash: str) -> Optional[Image]:
    """return the image row for this hash, or None if it's a new upload."""
    return session.query(Image).filter_by(sha256_hash=image_hash).first()


def find_cached_caption(
    session,
    image_id: int,
    model_variant: str,
    decoding_strategy: str,
    beam_width: Optional[int],
) -> Optional[Caption]:
    """
    return a previously generated Caption row matching this exact
    (image, model variant, decoding config) combination, or None if
    inference still needs to run.
    """
    return (
        session.query(Caption)
        .filter_by(
            image_id=image_id,
            model_variant=model_variant,
            decoding_strategy=decoding_strategy,
            beam_width=beam_width,
        )
        .first()
    )
