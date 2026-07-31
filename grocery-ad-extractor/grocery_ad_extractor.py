#!/usr/bin/env python3
"""Grocery Ad PDF Extractor.

Extracts products and prices from a grocery store advertisement PDF using
Mellea and Granite vision model.

Prerequisites:
    - Ollama running with: ollama pull hf.co/ibm-granite/granite-vision-4.1-4b-GGUF:Q4_K_M
    - Or use OpenAI backend with GPT-4 Vision (update backend initialization)
    - Install: pip install pydantic pdf2image pillow

Usage:
    python grocery_ad_extractor.py path/to/grocery_ad.pdf

Example:
    python grocery_ad_extractor.py HarrisTeeter.pdf
"""

import pathlib
import sys

from pdf2image import convert_from_path
from PIL import Image
from pydantic import BaseModel, Field

from mellea import start_session
from mellea.backends.model_options import ModelOption
from mellea.core import ImageBlock
from mellea.stdlib.context import ChatContext


class GroceryItem(BaseModel):
    """A single product from the grocery ad.

    Example:
        ```json
        {
          "product_name": "Organic Bananas",
          "price": "$0.59/lb",
          "unit": "lb",
          "discount": "Buy 2 Get 1 Free"
        }
        ```
    """

    product_name: str = Field(description="Name or description of the product")
    price: str = Field(description="Price as displayed (e.g., '$2.99' or '$4.99/lb')")
    unit: str | None = Field(
        default=None, description="Unit if applicable (e.g., 'lb', 'each', 'pack')"
    )
    discount: str | None = Field(
        default=None, description="Any discount or promotion (e.g., 'Buy 2 Get 1 Free')"
    )


class GroceryAdExtraction(BaseModel):
    """Extracted data from a grocery ad page.

    Example:
        ```json
        {
          "store_name": "Harris Teeter",
          "items": [
            {
              "product_name": "Organic Bananas",
              "price": "$0.59/lb",
              "unit": "lb",
              "discount": null
            }
          ],
          "notes": "Weekly specials valid through Sunday"
        }
        ```
    """

    items: list[GroceryItem] = Field(description="List of products found on this page")
    store_name: str | None = Field(
        default=None, description="Name of the grocery store if visible"
    )
    notes: str | None = Field(
        default=None, description="Any additional notes about the page"
    )


def extract_from_pdf(
    pdf_path: str, backend: str = "ollama", save_images: bool = True
) -> list[GroceryAdExtraction]:
    """Extract products and prices from a grocery ad PDF.

    Creates a separate session for each page to avoid context pollution between
    pages, ensuring each extraction is independent.

    Args:
        pdf_path: Path to the PDF file
        backend: Backend to use ("ollama" or "openai")
        save_images: Whether to save page images for debugging

    Returns:
        List of GroceryAdExtraction objects, one per PDF page
    """
    pdf_file = pathlib.Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Create debug directory for page images
    debug_dir = None
    if save_images:
        debug_dir = pathlib.Path("debug_pages")
        debug_dir.mkdir(exist_ok=True)
        print(f"Debug images will be saved to: {debug_dir}")

    # Convert PDF pages to images
    print(f"Converting PDF to images: {pdf_path}")
    images = convert_from_path(pdf_path)
    print(f"Extracted {len(images)} page(s)")

    # Initialize Mellea with larger context vision model
    model_id = "hf.co/ibm-granite/granite-vision-4.1-4b-GGUF:Q4_K_M"
    model_opts: dict[str, int] = {ModelOption.CONTEXT_WINDOW: 16384}  # Increased from 4096

    results = []

    for page_num, image in enumerate(images, 1):
        print(f"\nProcessing page {page_num}/{len(images)}...")

        # Save page image for debugging
        if save_images and debug_dir:
            image_path = debug_dir / f"page_{page_num:02d}.png"
            image.save(image_path)
            print(f"  Saved: {image_path}")

        try:
            # Create NEW session for each page to avoid context pollution
            with start_session(model_id=model_id, model_options=model_opts, ctx=ChatContext()) as m:
                # Convert PIL image to ImageBlock for Mellea
                img_block = ImageBlock.from_pil_image(image)

                # Create extraction prompt
                prompt = (
                    "Extract all grocery products with prices from this store advertisement. "
                    "For each product: name, price, unit (if shown), and any discounts.\n"
                    "The price like 2/$10 means $10 for 2. "
                    "Return structured JSON."
                )

                # Call vision model with structured output
                result = m.instruct(
                    prompt,
                    images=[img_block],
                    format=GroceryAdExtraction,
                )

                # Parse the JSON response
                extracted = GroceryAdExtraction.model_validate_json(str(result))
                results.append(extracted)

                # Print all items found on this page
                if extracted.store_name:
                    print(f"Store: {extracted.store_name}")
                print(f"✓ Found {len(extracted.items)} items")
                for item in extracted.items:
                    print(f"  - {item.product_name}: {item.price}")
                    if item.unit:
                        print(f"    Unit: {item.unit}")
                    if item.discount:
                        print(f"    Discount: {item.discount}")

        except Exception as e:
            print(f"✗ Error processing page {page_num}: {e}")
            continue

    return results


def export_to_csv(
    extractions: list[GroceryAdExtraction], output_path: str = "grocery_list.csv"
) -> None:
    """Export extracted items to a CSV file."""
    import csv

    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Page", "Store", "Product", "Price", "Unit", "Discount"])

            for page_num, extraction in enumerate(extractions, 1):
                for item in extraction.items:
                    writer.writerow(
                        [
                            page_num,
                            extraction.store_name or "",
                            item.product_name,
                            item.price,
                            item.unit or "",
                            item.discount or "",
                        ]
                    )

        print(f"\nExported to {output_path}")
    except OSError as e:
        print(f"✗ Error writing to {output_path}: {e}")
        raise


def export_to_markdown(
    extractions: list[GroceryAdExtraction], output_path: str = "grocery_list.md"
) -> None:
    """Export extracted items to a Markdown file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Grocery Ad Extraction\n\n")

            for page_num, extraction in enumerate(extractions, 1):
                f.write(f"## Page {page_num}\n")

                if extraction.store_name:
                    f.write(f"**Store:** {extraction.store_name}\n\n")

                if extraction.notes:
                    f.write(f"*Notes: {extraction.notes}*\n\n")

                f.write("| Product | Price | Unit | Discount |\n")
                f.write("|---------|-------|------|----------|\n")

                for item in extraction.items:
                    unit = item.unit or ""
                    discount = item.discount or ""
                    f.write(
                        f"| {item.product_name} | {item.price} | {unit} | {discount} |\n"
                    )

                f.write("\n")

        print(f"Exported to {output_path}")
    except OSError as e:
        print(f"✗ Error writing to {output_path}: {e}")
        raise


def main(pdf_path: str, backend: str = "ollama", save_images: bool = True) -> None:
    """Main entry point."""
    try:
        # Extract from PDF
        extractions = extract_from_pdf(pdf_path, backend, save_images=save_images)

        # Export results
        base_name = pathlib.Path(pdf_path).stem
        export_to_csv(extractions, f"{base_name}_output.csv")
        export_to_markdown(extractions, f"{base_name}_output.md")

        # Print summary
        total_items = sum(len(e.items) for e in extractions)
        print(f"\n✓ Extracted {total_items} items from {len(extractions)} page(s)")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ImportError as e:
        print(
            f"Error: Missing dependency: {e}\n"
            "Install with: uv pip install pdf2image pillow"
        )
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python grocery_ad_extractor.py <path_to_pdf> [backend] [--no-debug]")
        print("\nExample:")
        print("  python grocery_ad_extractor.py weekly_ad.pdf")
        print("  python grocery_ad_extractor.py weekly_ad.pdf ollama --no-debug")
        sys.exit(1)

    pdf_path = sys.argv[1]
    backend = sys.argv[2] if len(sys.argv) > 2 else "ollama"
    save_debug_images = "--no-debug" not in sys.argv
    main(pdf_path, backend, save_images=save_debug_images)
