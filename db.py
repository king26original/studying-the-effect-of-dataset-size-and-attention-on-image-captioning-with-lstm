"""
db.py
-----
Database layer for the caption workbench.

Uses SQLAlchemy on top of a local SQLite file (captions.db) — no server to
install or run, the database is just a file in this folder.

Two tables:
  - Image:   one row per UNIQUE uploaded image (deduplicated by content hash)
  - Caption: one row per generated caption (one image can have many —
             one per decoding config: model variant / strategy / beam width)
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------
DATABASE_URL = "sqlite:///captions.db"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True)
    sha256_hash = Column(String(64), unique=True, index=True, nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)  # where we saved it on disk
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    captions = relationship(
        "Caption", back_populates="image", cascade="all, delete-orphan"
    )


class Caption(Base):
    __tablename__ = "captions"

    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)

    model_variant = Column(String(50), nullable=False)       # "no_attention" | "attention"
    decoding_strategy = Column(String(50), nullable=False)   # "greedy" | "beam"
    beam_width = Column(Integer, nullable=True)               # null when strategy == "greedy"

    caption_text = Column(Text, nullable=False)
    latency_ms = Column(Float, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    image = relationship("Image", back_populates="captions")


def init_db():
    """Create tables if they don't exist yet. Safe to call on every app start."""
    Base.metadata.create_all(engine)


def get_session():
    return SessionLocal()