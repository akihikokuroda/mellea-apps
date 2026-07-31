"""Test suite for grocery_ad_extractor.py example.

pytest: unit, integration
"""

import json
import pathlib
import sys
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest
from PIL import Image
from pydantic import ValidationError

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from grocery_ad_extractor import (
    GroceryAdExtraction,
    GroceryItem,
    export_to_csv,
    export_to_markdown,
    extract_from_pdf,
)


class TestGroceryItem:
    """Unit tests for GroceryItem Pydantic model."""

    def test_minimal_item(self):
        """Construct minimal valid GroceryItem."""
        item = GroceryItem(product_name="Bananas", price="$0.59/lb")
        assert item.product_name == "Bananas"
        assert item.price == "$0.59/lb"
        assert item.unit is None
        assert item.discount is None

    def test_full_item(self):
        """Construct GroceryItem with all fields."""
        item = GroceryItem(
            product_name="Ground Beef",
            price="$5.99/lb",
            unit="lb",
            discount="Buy 2 Get 1 Free",
        )
        assert item.product_name == "Ground Beef"
        assert item.price == "$5.99/lb"
        assert item.unit == "lb"
        assert item.discount == "Buy 2 Get 1 Free"

    def test_missing_required_field_raises_validation_error(self):
        """Missing required field raises ValidationError."""
        with pytest.raises(ValidationError):
            GroceryItem(product_name="Bananas")

    def test_model_dump(self):
        """model_dump() returns dict representation."""
        item = GroceryItem(product_name="Milk", price="$3.99", unit="gallon")
        data = item.model_dump()
        assert data == {
            "product_name": "Milk",
            "price": "$3.99",
            "unit": "gallon",
            "discount": None,
        }


class TestGroceryAdExtraction:
    """Unit tests for GroceryAdExtraction Pydantic model."""

    def test_minimal_extraction(self):
        """Construct minimal valid extraction."""
        extraction = GroceryAdExtraction(items=[])
        assert extraction.items == []
        assert extraction.store_name is None
        assert extraction.notes is None

    def test_full_extraction(self):
        """Construct extraction with all fields."""
        items = [
            GroceryItem(product_name="Bananas", price="$0.59/lb"),
            GroceryItem(product_name="Milk", price="$3.99", unit="gallon"),
        ]
        extraction = GroceryAdExtraction(
            items=items, store_name="Harris Teeter", notes="Weekly specials"
        )
        assert len(extraction.items) == 2
        assert extraction.store_name == "Harris Teeter"
        assert extraction.notes == "Weekly specials"

    def test_model_validate_json(self):
        """Parse JSON string into GroceryAdExtraction."""
        json_str = json.dumps(
            {
                "items": [
                    {
                        "product_name": "Bananas",
                        "price": "$0.59/lb",
                        "unit": "lb",
                        "discount": None,
                    }
                ],
                "store_name": "Harris Teeter",
                "notes": None,
            }
        )
        extraction = GroceryAdExtraction.model_validate_json(json_str)
        assert len(extraction.items) == 1
        assert extraction.items[0].product_name == "Bananas"
        assert extraction.store_name == "Harris Teeter"

    def test_empty_items_list(self):
        """Extraction with empty items list is valid."""
        extraction = GroceryAdExtraction(items=[])
        assert extraction.items == []


class TestExportToCSV:
    """Integration tests for export_to_csv()."""

    def test_export_single_item(self, tmp_path):
        """Export single item to CSV."""
        csv_file = tmp_path / "output.csv"
        item = GroceryItem(product_name="Bananas", price="$0.59/lb", unit="lb")
        extraction = GroceryAdExtraction(items=[item], store_name="Harris Teeter")

        export_to_csv([extraction], str(csv_file))

        assert csv_file.exists()
        content = csv_file.read_text()
        assert "Page,Store,Product,Price,Unit,Discount" in content
        assert "1,Harris Teeter,Bananas,$0.59/lb,lb," in content

    def test_export_multiple_pages(self, tmp_path):
        """Export multiple pages to CSV."""
        csv_file = tmp_path / "output.csv"
        extractions = [
            GroceryAdExtraction(
                items=[GroceryItem(product_name="Bananas", price="$0.59/lb")],
                store_name="Harris Teeter",
            ),
            GroceryAdExtraction(
                items=[GroceryItem(product_name="Milk", price="$3.99")],
                store_name="Kroger",
            ),
        ]

        export_to_csv(extractions, str(csv_file))

        content = csv_file.read_text()
        assert "1,Harris Teeter,Bananas" in content
        assert "2,Kroger,Milk" in content

    def test_export_with_discounts(self, tmp_path):
        """Export item with discount."""
        csv_file = tmp_path / "output.csv"
        item = GroceryItem(
            product_name="Ground Beef",
            price="$5.99/lb",
            unit="lb",
            discount="Buy 2 Get 1 Free",
        )
        extraction = GroceryAdExtraction(items=[item], store_name="Safeway")

        export_to_csv([extraction], str(csv_file))

        content = csv_file.read_text()
        assert "Ground Beef,$5.99/lb,lb,Buy 2 Get 1 Free" in content

    def test_export_empty_extractions(self, tmp_path):
        """Export empty extractions list."""
        csv_file = tmp_path / "output.csv"
        export_to_csv([], str(csv_file))

        assert csv_file.exists()
        content = csv_file.read_text()
        # Should have header only
        assert "Page,Store,Product,Price,Unit,Discount" in content

    def test_export_permission_error(self, tmp_path):
        """Permission error raises OSError."""
        csv_file = tmp_path / "readonly" / "output.csv"
        csv_file.parent.mkdir()
        csv_file.parent.chmod(0o000)

        extraction = GroceryAdExtraction(
            items=[GroceryItem(product_name="Bananas", price="$0.59/lb")]
        )

        try:
            with pytest.raises(OSError):
                export_to_csv([extraction], str(csv_file))
        finally:
            csv_file.parent.chmod(0o755)


class TestExportToMarkdown:
    """Integration tests for export_to_markdown()."""

    def test_export_single_page(self, tmp_path):
        """Export single page to Markdown."""
        md_file = tmp_path / "output.md"
        item = GroceryItem(product_name="Bananas", price="$0.59/lb", unit="lb")
        extraction = GroceryAdExtraction(items=[item], store_name="Harris Teeter")

        export_to_markdown([extraction], str(md_file))

        assert md_file.exists()
        content = md_file.read_text()
        assert "# Grocery Ad Extraction" in content
        assert "## Page 1" in content
        assert "**Store:** Harris Teeter" in content
        assert "| Bananas | $0.59/lb | lb |" in content

    def test_export_with_notes(self, tmp_path):
        """Export page with notes."""
        md_file = tmp_path / "output.md"
        extraction = GroceryAdExtraction(
            items=[GroceryItem(product_name="Milk", price="$3.99")],
            store_name="Kroger",
            notes="Valid through Sunday",
        )

        export_to_markdown([extraction], str(md_file))

        content = md_file.read_text()
        assert "*Notes: Valid through Sunday*" in content

    def test_export_multiple_pages(self, tmp_path):
        """Export multiple pages to Markdown."""
        md_file = tmp_path / "output.md"
        extractions = [
            GroceryAdExtraction(
                items=[GroceryItem(product_name="Bananas", price="$0.59/lb")],
                store_name="Harris Teeter",
            ),
            GroceryAdExtraction(
                items=[GroceryItem(product_name="Milk", price="$3.99")],
                store_name="Kroger",
            ),
        ]

        export_to_markdown(extractions, str(md_file))

        content = md_file.read_text()
        assert "## Page 1" in content
        assert "## Page 2" in content
        assert "Harris Teeter" in content
        assert "Kroger" in content

    def test_export_with_discounts(self, tmp_path):
        """Export item with discount to Markdown."""
        md_file = tmp_path / "output.md"
        item = GroceryItem(
            product_name="Ground Beef",
            price="$5.99/lb",
            unit="lb",
            discount="Buy 2 Get 1 Free",
        )
        extraction = GroceryAdExtraction(items=[item])

        export_to_markdown([extraction], str(md_file))

        content = md_file.read_text()
        assert "| Ground Beef | $5.99/lb | lb | Buy 2 Get 1 Free |" in content

    def test_export_permission_error(self, tmp_path):
        """Permission error raises OSError."""
        md_file = tmp_path / "readonly" / "output.md"
        md_file.parent.mkdir()
        md_file.parent.chmod(0o000)

        extraction = GroceryAdExtraction(
            items=[GroceryItem(product_name="Bananas", price="$0.59/lb")]
        )

        try:
            with pytest.raises(OSError):
                export_to_markdown([extraction], str(md_file))
        finally:
            md_file.parent.chmod(0o755)


class TestExtractFromPDF:
    """Integration tests for extract_from_pdf() - mocked."""

    def test_pdf_not_found_raises_file_not_found_error(self):
        """Non-existent PDF raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            extract_from_pdf("/nonexistent/path/to/file.pdf")

    @patch("grocery_ad_extractor.convert_from_path")
    @patch("grocery_ad_extractor.start_session")
    def test_extract_single_page(self, mock_session, mock_convert):
        """Extract from single-page PDF."""
        # Create a mock image
        mock_image = Image.new("RGB", (100, 100), color="white")

        # Mock pdf2image
        mock_convert.return_value = [mock_image]

        # Mock Mellea session
        mock_m = MagicMock()
        mock_session.return_value.__enter__.return_value = mock_m

        # Mock extraction result
        extraction_data = {
            "items": [
                {
                    "product_name": "Bananas",
                    "price": "$0.59/lb",
                    "unit": "lb",
                    "discount": None,
                }
            ],
            "store_name": "Harris Teeter",
            "notes": None,
        }
        mock_result = MagicMock()
        mock_result.__str__ = MagicMock(return_value=json.dumps(extraction_data))
        mock_m.instruct.return_value = mock_result

        # Call extract_from_pdf with a fake PDF path
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            results = extract_from_pdf(tmp.name, save_images=False)

        assert len(results) == 1
        assert results[0].store_name == "Harris Teeter"
        assert len(results[0].items) == 1
        assert results[0].items[0].product_name == "Bananas"

    @patch("grocery_ad_extractor.convert_from_path")
    @patch("grocery_ad_extractor.start_session")
    def test_extract_multiple_pages(self, mock_session, mock_convert):
        """Extract from multi-page PDF."""
        # Create mock images
        mock_images = [Image.new("RGB", (100, 100), color="white") for _ in range(2)]
        mock_convert.return_value = mock_images

        # Mock Mellea session
        mock_m = MagicMock()
        mock_session.return_value.__enter__.return_value = mock_m

        # Mock extraction results for each page
        extraction_data_1 = {
            "items": [
                {
                    "product_name": "Bananas",
                    "price": "$0.59/lb",
                    "unit": "lb",
                    "discount": None,
                }
            ],
            "store_name": "Harris Teeter",
            "notes": None,
        }
        extraction_data_2 = {
            "items": [
                {
                    "product_name": "Milk",
                    "price": "$3.99",
                    "unit": None,
                    "discount": None,
                }
            ],
            "store_name": "Harris Teeter",
            "notes": None,
        }

        mock_results = [
            MagicMock(__str__=MagicMock(return_value=json.dumps(extraction_data_1))),
            MagicMock(__str__=MagicMock(return_value=json.dumps(extraction_data_2))),
        ]
        mock_m.instruct.side_effect = mock_results

        # Call extract_from_pdf
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            results = extract_from_pdf(tmp.name, save_images=False)

        assert len(results) == 2
        assert results[0].items[0].product_name == "Bananas"
        assert results[1].items[0].product_name == "Milk"

    @patch("grocery_ad_extractor.convert_from_path")
    @patch("grocery_ad_extractor.start_session")
    def test_extract_with_extraction_failure(self, mock_session, mock_convert):
        """Handle per-page extraction failure gracefully."""
        # Create mock images
        mock_images = [Image.new("RGB", (100, 100), color="white") for _ in range(2)]
        mock_convert.return_value = mock_images

        # Mock Mellea session - first succeeds, second fails
        mock_m = MagicMock()
        mock_session.return_value.__enter__.return_value = mock_m

        extraction_data = {
            "items": [
                {
                    "product_name": "Bananas",
                    "price": "$0.59/lb",
                    "unit": "lb",
                    "discount": None,
                }
            ],
            "store_name": "Harris Teeter",
            "notes": None,
        }
        mock_result = MagicMock(
            __str__=MagicMock(return_value=json.dumps(extraction_data))
        )

        # First call succeeds, second raises exception
        mock_m.instruct.side_effect = [mock_result, Exception("Extraction failed")]

        # Call extract_from_pdf
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            results = extract_from_pdf(tmp.name, save_images=False)

        # Should have one successful result, one skipped
        assert len(results) == 1
        assert results[0].items[0].product_name == "Bananas"

    @patch("grocery_ad_extractor.convert_from_path")
    def test_extract_with_debug_images(self, mock_convert, tmp_path):
        """Save debug images when save_images=True."""
        mock_image = Image.new("RGB", (100, 100), color="white")
        mock_convert.return_value = [mock_image]

        debug_dir = tmp_path / "debug_pages"

        with patch("grocery_ad_extractor.start_session") as mock_session:
            mock_m = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_m

            extraction_data = {
                "items": [
                    {
                        "product_name": "Bananas",
                        "price": "$0.59/lb",
                        "unit": "lb",
                        "discount": None,
                    }
                ],
                "store_name": "Harris Teeter",
                "notes": None,
            }
            mock_result = MagicMock(
                __str__=MagicMock(return_value=json.dumps(extraction_data))
            )
            mock_m.instruct.return_value = mock_result

            with tempfile.NamedTemporaryFile(suffix=".pdf", dir=tmp_path) as tmp:
                # Change to tmp_path so debug_pages is created there
                original_cwd = pathlib.Path.cwd()
                try:
                    import os

                    os.chdir(tmp_path)
                    extract_from_pdf(tmp.name, save_images=True)
                finally:
                    os.chdir(original_cwd)

            # Check debug images were created
            assert debug_dir.exists()
            assert len(list(debug_dir.glob("page_*.png"))) == 1
