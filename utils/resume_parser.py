"""Simple PDF resume text extraction and skill parsing."""

from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader

from utils.text_processing import extract_keywords


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def extract_skills_from_resume(file_bytes: bytes) -> list[str]:
    text = extract_text_from_pdf(file_bytes)
    return extract_keywords(text)
