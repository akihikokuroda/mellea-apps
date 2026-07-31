#!/usr/bin/env python3
"""Advanced Grocery Ad PDF Extractor.

Enhanced version with requirements validation, sampling strategies, and
better error handling.

Prerequisites:
    - Ollama running with: ollama pull granite3.2-vision
    - Install: uv pip install pydantic pdf2image pillow

Usage:
    uv run python grocery_ad_extractor_advanced.py path/to/grocery_ad.pdf
"""

import json
import pathlib
import sys

from pdf2image import convert_from_path
from PIL import Image
from pydantic import BaseModel, Field

from mellea import start_session
from mellea.backends.model_options import ModelOption
from mellea.core import ImageBlock, Requirement
from mellea.stdlib.context import ChatContext


class GroceryItem(BaseModel):
    """A single product from the grocery ad."""

    product_name: str = Field(description="Name or description of the product")
    price: str = Field(description="Price as displayed (e.g., '$2.99')")
    unit: str | None = Field(default=None, description="Unit (lb, each, pack)")
    original_price: str | None = Field(
        default=None, description="Original price if on sale"
    )
    discount_percent: int | None = Field(
        default=None, description="Discount percentage"
    )
    discount_description: str | None = Field(default=None, description="Promotion text")


class GroceryAdExtraction(BaseModel):
    """Extracted data from a grocery ad page."""

    page_number: int = Field(description="Page number in the PDF")
    items: list[GroceryItem] = Field(description="List of products")
    store_name: str | None = Field(default=None, description="Store name if visible")
    page_date: str | None = Field(
        default=None, description="Valid date range if visible"
    )
    confidence: float = Field(
        default=1.0, description="Confidence score for extraction (0-1)", ge=0, le=1
    )
    extraction_notes: str | None = Field(
        default=None, description="Notes about extraction"
    )


class GroceryAdExtractorAdvanced:
    """Advanced extractor with validation and retry logic."""

    def __init__(
        self,
        model_id: str = "hf.co/ibm-granite/granite-vision-4.1-4b-GGUF:Q4_K_M",
        max_retries: int = 2,
    ):
        """Initialize the extractor.

        Args:
            model_id: Vision model to use (uses larger context Granite by default)
            max_retries: Number of retries for failed extractions
        """
        self.model_id = model_id
        self.max_retries = max_retries

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        return None

    def extract_page(
        self, image, page_num: int, validate: bool = True
    ) -> GroceryAdExtraction | None:
        """Extract products from a single page image.

        Args:
            image: PIL Image object
            page_num: Page number
            validate: Whether to validate extraction with requirements

        Returns:
            GroceryAdExtraction or None if extraction fails
        """
        prompt = (
            "Extract all products and prices from this grocery ad.\n"
            "For each item: product_name, price, unit, discounts.\n"
            "Also identify store_name if visible."
        )

        # Requirements for validation
        requirements = (
            [
                Requirement("All prices must be in currency format (e.g., $X.XX)"),
                Requirement("Product names must be non-empty"),
                Requirement("At least one product must be extracted"),
            ]
            if validate
            else []
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                # Create fresh session for each page (context isolation)
                model_opts: dict[str, int] = {ModelOption.CONTEXT_WINDOW: 16384}
                with start_session(
                    model_id=self.model_id, model_options=model_opts, ctx=ChatContext()
                ) as session:
                    # Convert PIL image to ImageBlock
                    img_block = ImageBlock.from_pil_image(image)

                    result = session.instruct(
                        prompt,
                        images=[img_block],
                        format=GroceryAdExtraction,
                        requirements=requirements if validate else None,
                    )

                    extraction = GroceryAdExtraction.model_validate_json(str(result))
                    extraction.page_number = page_num
                    extraction.confidence = 1.0 if attempt == 1 else 0.8

                    return extraction

            except Exception as e:
                if attempt == self.max_retries:
                    print(
                        f"Failed to extract page {page_num} after {self.max_retries} retries: {e}"
                    )
                    return None
                print(f"Attempt {attempt} failed for page {page_num}, retrying...")

        return None

    def extract_pdf(
        self, pdf_path: str, validate: bool = True, save_images: bool = True
    ) -> list[GroceryAdExtraction]:
        """Extract products from all pages in a PDF.

        Args:
            pdf_path: Path to PDF file
            validate: Whether to validate with requirements
            save_images: Whether to save page images for debugging

        Returns:
            List of GroceryAdExtraction objects
        """
        pdf_file = pathlib.Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Create debug directory for page images
        debug_dir = None
        if save_images:
            debug_dir = pathlib.Path("debug_pages")
            debug_dir.mkdir(exist_ok=True)
            print(f"Debug images will be saved to: {debug_dir}\n")

        print(f"Converting PDF to images: {pdf_path}")
        images = convert_from_path(pdf_path)
        print(f"Extracted {len(images)} page(s)\n")

        results = []
        for page_num, image in enumerate(images, 1):
            # Save page image for debugging
            if save_images and debug_dir:
                image_path = debug_dir / f"page_{page_num:02d}.png"
                image.save(image_path)

            print(f"[{page_num}/{len(images)}] Processing page {page_num}...", end=" ")

            extraction = self.extract_page(image, page_num, validate=validate)

            if extraction:
                results.append(extraction)
                print(f"✓ Found {len(extraction.items)} items")
                for item in extraction.items:
                    print(f"  - {item.product_name}: {item.price}")
                    if item.unit:
                        print(f"    Unit: {item.unit}")
                    if item.discount_description:
                        print(f"    Discount: {item.discount_description}")
            else:
                print("✗ Extraction failed")

        return results


def export_results(
    extractions: list[GroceryAdExtraction], output_dir: str | None = None
) -> None:
    """Export results in multiple formats.

    Args:
        extractions: List of extraction results
        output_dir: Output directory (default: current directory)
    """
    import csv

    output_path = pathlib.Path(output_dir or ".")
    output_path.mkdir(parents=True, exist_ok=True)

    # Export as JSON
    json_path = output_path / "extraction_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([e.model_dump() for e in extractions], f, indent=2, default=str)
    print(f"✓ JSON export: {json_path}")

    # Export as CSV
    csv_path = output_path / "grocery_items.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Page",
                "Store",
                "Product",
                "Price",
                "Unit",
                "Original Price",
                "Discount %",
                "Promotion",
                "Confidence",
            ]
        )

        for extraction in extractions:
            for item in extraction.items:
                writer.writerow(
                    [
                        extraction.page_number,
                        extraction.store_name or "",
                        item.product_name,
                        item.price,
                        item.unit or "",
                        item.original_price or "",
                        item.discount_percent or "",
                        item.discount_description or "",
                        extraction.confidence,
                    ]
                )

    print(f"✓ CSV export: {csv_path}")

    # Print summary
    total_items = sum(len(e.items) for e in extractions)
    print(f"\nSummary: {total_items} items from {len(extractions)} page(s)")
    if extractions:
        avg_confidence = sum(e.confidence for e in extractions) / len(extractions)
        print(f"Average confidence: {avg_confidence:.2f}")


def main(pdf_path: str, save_images: bool = True):
    """Main entry point."""
    try:
        with GroceryAdExtractorAdvanced() as extractor:
            extractions = extractor.extract_pdf(
                pdf_path, validate=True, save_images=save_images
            )

        if extractions:
            export_results(extractions)
        else:
            print("No products extracted from PDF")
            sys.exit(1)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"Error: Missing dependency: {e}")
        print("Install with: pip install pdf2image pillow")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python grocery_ad_extractor_advanced.py <path_to_pdf> [--no-debug]"
        )
        print("\nExample:")
        print("  python grocery_ad_extractor_advanced.py weekly_ad.pdf")
        print("  python grocery_ad_extractor_advanced.py weekly_ad.pdf --no-debug")
        sys.exit(1)

    pdf_path = sys.argv[1]
    save_debug_images = "--no-debug" not in sys.argv
    main(pdf_path, save_images=save_debug_images)
