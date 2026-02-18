"""
PDF parser using the `unstructured` library.

Extracts structured elements (text, tables, images) from PDF files
using the hi_res strategy for maximum fidelity.
"""
import os
import glob
from unstructured.partition.pdf import partition_pdf

from src.app.logger import get_logger
from src.app.utils import timer

logger = get_logger(__name__)


@timer
def parse_pdf(file_path: str, extract_images: bool = True) -> list:
    """
    Parse a single PDF file and extract structured elements.

    Args:
        file_path: Path to the PDF file.
        extract_images: Whether to extract images from the PDF.

    Returns:
        List of unstructured Element objects.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    logger.info(f"📃 Partitioning document: {file_path}")

    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        extract_images_in_pdf=extract_images,
        extract_image_block_to_payload=extract_images,
        infer_table_structure=True,
    )

    logger.info(f"✅ Extracted {len(elements)} elements from {os.path.basename(file_path)}")
    return elements


def parse_directory(dir_path: str, extract_images: bool = True) -> dict:
    """
    Parse all PDF files in a directory.

    Args:
        dir_path: Path to the directory containing PDFs.
        extract_images: Whether to extract images from PDFs.

    Returns:
        Dictionary mapping filename -> list of elements.
    """
    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"Directory not found: {dir_path}")

    pdf_files = glob.glob(os.path.join(dir_path, "*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {dir_path}")
        return {}

    logger.info(f"📂 Found {len(pdf_files)} PDF(s) in {dir_path}")

    results = {}
    for pdf_path in sorted(pdf_files):
        filename = os.path.basename(pdf_path)
        try:
            elements = parse_pdf(pdf_path, extract_images=extract_images)
            results[filename] = elements
        except Exception as e:
            logger.error(f"Failed to parse {filename}: {e}")

    return results
