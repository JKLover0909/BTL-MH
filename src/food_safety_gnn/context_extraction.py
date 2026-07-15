"""Native PDF extraction and optional Tesseract OCR comparison."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz
import pytesseract
from PIL import Image

from food_safety_gnn.provenance import atomic_write_json, atomic_write_text, sha256_file, utc_timestamp


class ContextExtractionError(RuntimeError):
    """Raised when native extraction or requested OCR validation cannot complete."""


@dataclass(frozen=True)
class TesseractInfo:
    """Resolved Tesseract executable metadata."""

    executable: str
    version: str
    languages: list[str]


def normalize_text(text: str) -> str:
    """Normalize whitespace before a transparent native/OCR comparison."""
    return re.sub(r"\s+", " ", text).strip()


def resolve_tesseract() -> TesseractInfo:
    """Resolve, validate, and describe Tesseract using an override or PATH."""
    configured = os.environ.get("TESSERACT_CMD")
    executable = configured or shutil.which("tesseract")
    if not executable:
        raise ContextExtractionError(
            "Tesseract was not found. Install it in meibook-dev or set TESSERACT_CMD to an "
            "executable path."
        )

    executable_path = Path(executable).expanduser().resolve()
    if not executable_path.is_file() or not os.access(executable_path, os.X_OK):
        raise ContextExtractionError(f"TESSERACT_CMD is not an executable file: {executable_path}")

    try:
        version_result = subprocess.run(
            [str(executable_path), "--version"], check=True, capture_output=True, text=True
        )
        language_result = subprocess.run(
            [str(executable_path), "--list-langs"], check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as error:
        raise ContextExtractionError(f"Unable to run Tesseract at {executable_path}: {error.stderr}") from error

    pytesseract.pytesseract.tesseract_cmd = str(executable_path)
    try:
        pytesseract_version = str(pytesseract.get_tesseract_version())
    except pytesseract.TesseractNotFoundError as error:
        raise ContextExtractionError(
            f"pytesseract cannot execute {executable_path}; verify dependencies and TESSERACT_CMD."
        ) from error

    language_lines = [line.strip() for line in language_result.stdout.splitlines() if line.strip()]
    languages = [line for line in language_lines if not line.startswith("List of available languages")]
    return TesseractInfo(
        executable=str(executable_path),
        version=f"{version_result.stdout.splitlines()[0]} | pytesseract sees {pytesseract_version}",
        languages=languages,
    )


def extract_context(
    source_pdf: Path,
    output_markdown: Path,
    provenance_path: Path,
    ocr_enabled: bool = True,
    language: str = "eng",
    dpi: int = 200,
) -> dict[str, object]:
    """Extract each PDF page and optionally compare it to Tesseract OCR."""
    if not source_pdf.is_file():
        raise ContextExtractionError(f"PDF input does not exist: {source_pdf}")
    if dpi < 72:
        raise ContextExtractionError("DPI must be at least 72 for readable rasterization.")

    tesseract = resolve_tesseract() if ocr_enabled else None
    if tesseract and language not in tesseract.languages:
        raise ContextExtractionError(
            f"Tesseract language '{language}' is unavailable. Installed languages include: "
            f"{', '.join(tesseract.languages[:20])}."
        )

    source_hash = sha256_file(source_pdf)
    document = fitz.open(source_pdf)
    page_records: list[dict[str, object]] = []
    markdown_pages: list[str] = []

    try:
        for page_number, page in enumerate(document, start=1):
            native_text = page.get_text("text").strip()
            record: dict[str, object] = {
                "page": page_number,
                "native_characters": len(native_text),
                "native_text_sha256": __import__("hashlib").sha256(
                    native_text.encode("utf-8")
                ).hexdigest(),
                "ocr_status": "not_run" if not ocr_enabled else "completed",
            }
            page_sections = [f"## Page {page_number}", "", "### Native PDF text", "", native_text]

            if tesseract:
                scale = dpi / 72
                pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                ocr_text = pytesseract.image_to_string(image, lang=language).strip()
                native_normalized = normalize_text(native_text)
                ocr_normalized = normalize_text(ocr_text)
                matching_characters = sum(
                    left == right for left, right in zip(native_normalized, ocr_normalized)
                )
                denominator = max(len(native_normalized), len(ocr_normalized), 1)
                record.update(
                    {
                        "ocr_characters": len(ocr_text),
                        "ocr_text_sha256": __import__("hashlib").sha256(
                            ocr_text.encode("utf-8")
                        ).hexdigest(),
                        "normalized_prefix_character_agreement": round(
                            matching_characters / denominator, 4
                        ),
                        "ocr_settings": {"language": language, "dpi": dpi, "psm": 3},
                    }
                )
                page_sections.extend(["", "### Tesseract OCR text", "", ocr_text])

            page_records.append(record)
            markdown_pages.append("\n".join(page_sections).rstrip())
    finally:
        document.close()

    metadata: dict[str, object] = {
        "source": str(source_pdf.resolve()),
        "source_sha256": source_hash,
        "page_count": len(page_records),
        "extracted_at": utc_timestamp(),
        "native_engine": f"PyMuPDF {fitz.VersionBind}",
        "ocr_requested": ocr_enabled,
        "ocr_validation_status": "pending_human_review" if ocr_enabled else "ocr_not_run",
        "tesseract": asdict(tesseract) if tesseract else None,
        "pages": page_records,
    }
    front_matter = "---\n" + "\n".join(
        f"{key}: {value}" for key, value in metadata.items() if key != "pages"
    ) + "\n---"
    content = "\n\n".join(
        [
            front_matter,
            "# Context extracted from X.pdf",
            "",
            "Native PDF text is preserved page-by-page. OCR is a comparison artifact and remains "
            "pending human review; it is not an authoritative transcription.",
            "",
            *markdown_pages,
            "",
        ]
    )
    atomic_write_text(output_markdown, content)
    metadata["output"] = str(output_markdown.resolve())
    metadata["output_sha256"] = sha256_file(output_markdown)
    atomic_write_json(provenance_path, metadata)
    return metadata
