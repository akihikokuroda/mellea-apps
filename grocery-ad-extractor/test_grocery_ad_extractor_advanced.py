"""Test suite for grocery_ad_extractor_advanced.py example.

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

from grocery_ad_extractor_advanced import (
    GroceryAdExtraction,
    GroceryAdExtractorAdvanced,
    GroceryItem,
    export_results,
)


class TestGroceryItemAdvanced:
    """Unit tests for advanced GroceryItem Pydantic model."""

    def test_minimal_item(self):
        """Construct minimal valid GroceryItem."""
        item = GroceryItem(product_name="Bananas", price="$0.59/lb")
        assert item.product_name == "Bananas"
        assert item.price == "$0.59/lb"
        assert item.unit is None
        assert item.original_price is None
        assert item.discount_percent is None
        assert item.discount_description is None

    def test_full_item_with_discount_percent(self):
        """Construct GroceryItem with all fields including discount percent."""
        item = GroceryItem(
            product_name="Ground Beef",
            price="$4.99/lb",
            unit="lb",
            original_price="$5.99/lb",
            discount_percent=17,
            discount_description="Buy 2 Get 1 Free",
        )
        assert item.product_name == "Ground Beef"
        assert item.price == "$4.99/lb"
        assert item.unit == "lb"
        assert item.original_price == "$5.99/lb"
        assert item.discount_percent == 17
        assert item.discount_description == "Buy 2 Get 1 Free"

    def test_missing_required_field_raises_validation_error(self):
        """Missing required field raises ValidationError."""
        with pytest.raises(ValidationError):
            GroceryItem(product_name="Bananas")

    def test_model_dump_with_all_fields(self):
        """model_dump() returns complete dict representation."""
        item = GroceryItem(
            product_name="Milk",
            price="$3.99",
            unit="gallon",
            original_price="$4.49",
            discount_percent=11,
            discount_description="Weekly special",
        )
        data = item.model_dump()
        assert data["product_name"] == "Milk"
        assert data["original_price"] == "$4.49"
        assert data["discount_percent"] == 11


class TestGroceryAdExtractionAdvanced:
    """Unit tests for advanced GroceryAdExtraction Pydantic model."""

    def test_minimal_extraction(self):
        """Construct minimal valid extraction."""
        extraction = GroceryAdExtraction(page_number=1, items=[])
        assert extraction.page_number == 1
        assert extraction.items == []
        assert extraction.confidence == 1.0
        assert extraction.store_name is None
        assert extraction.page_date is None
        assert extraction.extraction_notes is None

    def test_full_extraction(self):
        """Construct extraction with all fields."""
        items = [
            GroceryItem(product_name="Bananas", price="$0.59/lb", unit="lb"),
            GroceryItem(
                product_name="Milk", price="$3.99", unit="gallon", discount_percent=10
            ),
        ]
        extraction = GroceryAdExtraction(
            page_number=2,
            items=items,
            store_name="Harris Teeter",
            page_date="Valid 1/15-1/21",
            confidence=0.95,
            extraction_notes="High quality scan",
        )
        assert extraction.page_number == 2
        assert len(extraction.items) == 2
        assert extraction.store_name == "Harris Teeter"
        assert extraction.page_date == "Valid 1/15-1/21"
        assert extraction.confidence == 0.95
        assert extraction.extraction_notes == "High quality scan"

    def test_confidence_validation_range(self):
        """Confidence must be between 0 and 1."""
        # Valid: 0.0
        extraction = GroceryAdExtraction(page_number=1, items=[], confidence=0.0)
        assert extraction.confidence == 0.0

        # Valid: 1.0
        extraction = GroceryAdExtraction(page_number=1, items=[], confidence=1.0)
        assert extraction.confidence == 1.0

        # Invalid: > 1.0
        with pytest.raises(ValidationError):
            GroceryAdExtraction(page_number=1, items=[], confidence=1.5)

        # Invalid: < 0.0
        with pytest.raises(ValidationError):
            GroceryAdExtraction(page_number=1, items=[], confidence=-0.1)

    def test_model_validate_json(self):
        """Parse JSON string into GroceryAdExtraction."""
        json_str = json.dumps(
            {
                "page_number": 1,
                "items": [
                    {
                        "product_name": "Bananas",
                        "price": "$0.59/lb",
                        "unit": "lb",
                        "original_price": None,
                        "discount_percent": None,
                        "discount_description": None,
                    }
                ],
                "store_name": "Harris Teeter",
                "page_date": None,
                "confidence": 1.0,
                "extraction_notes": None,
            }
        )
        extraction = GroceryAdExtraction.model_validate_json(json_str)
        assert extraction.page_number == 1
        assert len(extraction.items) == 1
        assert extraction.items[0].product_name == "Bananas"
        assert extraction.store_name == "Harris Teeter"


class TestGroceryAdExtractorAdvanced:
    """Unit tests for GroceryAdExtractorAdvanced class."""

    def test_initialization(self):
        """Initialize extractor with default parameters."""
        extractor = GroceryAdExtractorAdvanced()
        assert (
            extractor.model_id == "hf.co/ibm-granite/granite-vision-4.1-4b-GGUF:Q4_K_M"
        )
        assert extractor.max_retries == 2

    def test_initialization_custom_params(self):
        """Initialize extractor with custom parameters."""
        extractor = GroceryAdExtractorAdvanced(model_id="gpt-4-vision", max_retries=3)
        assert extractor.model_id == "gpt-4-vision"
        assert extractor.max_retries == 3

    def test_context_manager_entry(self):
        """Context manager __enter__ returns self."""
        extractor = GroceryAdExtractorAdvanced()
        assert extractor.__enter__() is extractor

    def test_context_manager_exit(self):
        """Context manager __exit__ returns None."""
        extractor = GroceryAdExtractorAdvanced()
        result = extractor.__exit__(None, None, None)
        assert result is None

    def test_context_manager_protocol(self):
        """Use extractor as context manager."""
        with GroceryAdExtractorAdvanced() as extractor:
            assert isinstance(extractor, GroceryAdExtractorAdvanced)

    @patch("grocery_ad_extractor_advanced.start_session")
    def test_extract_page_success(self, mock_session):
        """Successful page extraction."""
        extractor = GroceryAdExtractorAdvanced()
        mock_image = Image.new("RGB", (100, 100), color="white")

        # Mock session
        mock_m = MagicMock()
        mock_session.return_value.__enter__.return_value = mock_m

        extraction_data = {
            "page_number": 0,  # Will be overwritten
            "items": [
                {
                    "product_name": "Bananas",
                    "price": "$0.59/lb",
                    "unit": "lb",
                    "original_price": None,
                    "discount_percent": None,
                    "discount_description": None,
                }
            ],
            "store_name": "Harris Teeter",
            "page_date": None,
            "confidence": 1.0,
            "extraction_notes": None,
        }
        mock_result = MagicMock(
            __str__=MagicMock(return_value=json.dumps(extraction_data))
        )
        mock_m.instruct.return_value = mock_result

        result = extractor.extract_page(mock_image, page_num=1, validate=True)

        assert result is not None
        assert result.page_number == 1
        assert result.confidence == 1.0
        assert len(result.items) == 1
        assert result.items[0].product_name == "Bananas"

    @patch("grocery_ad_extractor_advanced.start_session")
    def test_extract_page_retry_reduces_confidence(self, mock_session):
        """Retry attempt reduces confidence score."""
        extractor = GroceryAdExtractorAdvanced(max_retries=2)
        mock_image = Image.new("RGB", (100, 100), color="white")

        # Mock session
        mock_m = MagicMock()
        mock_session.return_value.__enter__.return_value = mock_m

        extraction_data = {
            "page_number": 0,
            "items": [
                {
                    "product_name": "Bananas",
                    "price": "$0.59/lb",
                    "unit": "lb",
                    "original_price": None,
                    "discount_percent": None,
                    "discount_description": None,
                }
            ],
            "store_name": "Harris Teeter",
            "page_date": None,
            "confidence": 1.0,
            "extraction_notes": None,
        }
        mock_result = MagicMock(
            __str__=MagicMock(return_value=json.dumps(extraction_data))
        )

        # First call raises exception, second succeeds
        mock_m.instruct.side_effect = [Exception("First attempt failed"), mock_result]

        result = extractor.extract_page(mock_image, page_num=1, validate=True)

        # Second attempt (retry) should have lower confidence
        assert result is not None
        assert result.confidence == 0.8

    @patch("grocery_ad_extractor_advanced.start_session")
    def test_extract_page_all_retries_exhausted(self, mock_session):
        """Return None when all retries exhausted."""
        extractor = GroceryAdExtractorAdvanced(max_retries=2)
        mock_image = Image.new("RGB", (100, 100), color="white")

        # Mock session
        mock_m = MagicMock()
        mock_session.return_value.__enter__.return_value = mock_m

        # All attempts fail
        mock_m.instruct.side_effect = Exception("Extraction failed")

        result = extractor.extract_page(mock_image, page_num=1, validate=True)

        assert result is None

    @patch("grocery_ad_extractor_advanced.start_session")
    def test_extract_page_with_validation_disabled(self, mock_session):
        """Extract without validation requirements."""
        extractor = GroceryAdExtractorAdvanced()
        mock_image = Image.new("RGB", (100, 100), color="white")

        # Mock session
        mock_m = MagicMock()
        mock_session.return_value.__enter__.return_value = mock_m

        extraction_data = {
            "page_number": 0,
            "items": [
                {
                    "product_name": "Bananas",
                    "price": "$0.59/lb",
                    "unit": "lb",
                    "original_price": None,
                    "discount_percent": None,
                    "discount_description": None,
                }
            ],
            "store_name": "Harris Teeter",
            "page_date": None,
            "confidence": 1.0,
            "extraction_notes": None,
        }
        mock_result = MagicMock(
            __str__=MagicMock(return_value=json.dumps(extraction_data))
        )
        mock_m.instruct.return_value = mock_result

        result = extractor.extract_page(mock_image, page_num=1, validate=False)

        # Verify requirements=None was passed
        call_kwargs = mock_m.instruct.call_args[1]
        assert call_kwargs["requirements"] is None
        assert result is not None

    @patch("grocery_ad_extractor_advanced.convert_from_path")
    def test_extract_pdf_not_found(self, mock_convert):
        """Non-existent PDF raises FileNotFoundError."""
        extractor = GroceryAdExtractorAdvanced()
        with pytest.raises(FileNotFoundError):
            extractor.extract_pdf("/nonexistent/file.pdf")

    @patch("grocery_ad_extractor_advanced.convert_from_path")
    @patch("grocery_ad_extractor_advanced.start_session")
    def test_extract_pdf_single_page(self, mock_session, mock_convert):
        """Extract from single-page PDF."""
        extractor = GroceryAdExtractorAdvanced()
        mock_image = Image.new("RGB", (100, 100), color="white")
        mock_convert.return_value = [mock_image]

        # Mock session
        mock_m = MagicMock()
        mock_session.return_value.__enter__.return_value = mock_m

        extraction_data = {
            "page_number": 0,
            "items": [
                {
                    "product_name": "Bananas",
                    "price": "$0.59/lb",
                    "unit": "lb",
                    "original_price": None,
                    "discount_percent": None,
                    "discount_description": None,
                }
            ],
            "store_name": "Harris Teeter",
            "page_date": None,
            "confidence": 1.0,
            "extraction_notes": None,
        }
        mock_result = MagicMock(
            __str__=MagicMock(return_value=json.dumps(extraction_data))
        )
        mock_m.instruct.return_value = mock_result

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            results = extractor.extract_pdf(tmp.name, validate=True, save_images=False)

        assert len(results) == 1
        assert results[0].page_number == 1
        assert results[0].store_name == "Harris Teeter"

    @patch("grocery_ad_extractor_advanced.convert_from_path")
    @patch("grocery_ad_extractor_advanced.start_session")
    def test_extract_pdf_multiple_pages(self, mock_session, mock_convert):
        """Extract from multi-page PDF."""
        extractor = GroceryAdExtractorAdvanced()
        mock_images = [Image.new("RGB", (100, 100), color="white") for _ in range(2)]
        mock_convert.return_value = mock_images

        # Mock session
        mock_m = MagicMock()
        mock_session.return_value.__enter__.return_value = mock_m

        extraction_data_1 = {
            "page_number": 0,
            "items": [
                {
                    "product_name": "Bananas",
                    "price": "$0.59/lb",
                    "unit": "lb",
                    "original_price": None,
                    "discount_percent": None,
                    "discount_description": None,
                }
            ],
            "store_name": "Harris Teeter",
            "page_date": None,
            "confidence": 1.0,
            "extraction_notes": None,
        }
        extraction_data_2 = {
            "page_number": 0,
            "items": [
                {
                    "product_name": "Milk",
                    "price": "$3.99",
                    "unit": "gallon",
                    "original_price": None,
                    "discount_percent": None,
                    "discount_description": None,
                }
            ],
            "store_name": "Harris Teeter",
            "page_date": None,
            "confidence": 1.0,
            "extraction_notes": None,
        }

        mock_results = [
            MagicMock(__str__=MagicMock(return_value=json.dumps(extraction_data_1))),
            MagicMock(__str__=MagicMock(return_value=json.dumps(extraction_data_2))),
        ]
        mock_m.instruct.side_effect = mock_results

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            results = extractor.extract_pdf(tmp.name, validate=True, save_images=False)

        assert len(results) == 2
        assert results[0].page_number == 1
        assert results[1].page_number == 2
        assert results[0].items[0].product_name == "Bananas"
        assert results[1].items[0].product_name == "Milk"

    @patch("grocery_ad_extractor_advanced.convert_from_path")
    @patch("grocery_ad_extractor_advanced.start_session")
    def test_extract_pdf_partial_failure(self, mock_session, mock_convert):
        """Handle per-page failures gracefully."""
        extractor = GroceryAdExtractorAdvanced()
        mock_images = [Image.new("RGB", (100, 100), color="white") for _ in range(2)]
        mock_convert.return_value = mock_images

        # Mock session
        mock_m = MagicMock()
        mock_session.return_value.__enter__.return_value = mock_m

        extraction_data = {
            "page_number": 0,
            "items": [
                {
                    "product_name": "Bananas",
                    "price": "$0.59/lb",
                    "unit": "lb",
                    "original_price": None,
                    "discount_percent": None,
                    "discount_description": None,
                }
            ],
            "store_name": "Harris Teeter",
            "page_date": None,
            "confidence": 1.0,
            "extraction_notes": None,
        }
        mock_result = MagicMock(
            __str__=MagicMock(return_value=json.dumps(extraction_data))
        )

        # First succeeds, second fails
        mock_m.instruct.side_effect = [mock_result, Exception("Failed")]

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            results = extractor.extract_pdf(tmp.name, validate=True, save_images=False)

        # Should have one successful result
        assert len(results) == 1
        assert results[0].items[0].product_name == "Bananas"


class TestExportResults:
    """Integration tests for export_results()."""

    def test_export_json_single_extraction(self, tmp_path):
        """Export single extraction to JSON."""
        extraction = GroceryAdExtraction(
            page_number=1,
            items=[GroceryItem(product_name="Bananas", price="$0.59/lb", unit="lb")],
            store_name="Harris Teeter",
            confidence=1.0,
        )

        export_results([extraction], str(tmp_path))

        json_file = tmp_path / "extraction_results.json"
        assert json_file.exists()

        data = json.loads(json_file.read_text())
        assert len(data) == 1
        assert data[0]["page_number"] == 1
        assert data[0]["store_name"] == "Harris Teeter"
        assert len(data[0]["items"]) == 1
        assert data[0]["items"][0]["product_name"] == "Bananas"

    def test_export_csv_single_extraction(self, tmp_path):
        """Export single extraction to CSV."""
        extraction = GroceryAdExtraction(
            page_number=1,
            items=[
                GroceryItem(
                    product_name="Ground Beef",
                    price="$5.99/lb",
                    unit="lb",
                    discount_percent=15,
                    discount_description="Buy 2 Get 1 Free",
                )
            ],
            store_name="Safeway",
            confidence=1.0,
        )

        export_results([extraction], str(tmp_path))

        csv_file = tmp_path / "grocery_items.csv"
        assert csv_file.exists()

        content = csv_file.read_text()
        assert (
            "Page,Store,Product,Price,Unit,Original Price,Discount %,Promotion,Confidence"
            in content
        )
        assert "1,Safeway,Ground Beef,$5.99/lb,lb,,15,Buy 2 Get 1 Free,1.0" in content

    def test_export_multiple_extractions(self, tmp_path):
        """Export multiple extractions."""
        extractions = [
            GroceryAdExtraction(
                page_number=1,
                items=[
                    GroceryItem(product_name="Bananas", price="$0.59/lb", unit="lb")
                ],
                store_name="Harris Teeter",
                confidence=1.0,
            ),
            GroceryAdExtraction(
                page_number=2,
                items=[GroceryItem(product_name="Milk", price="$3.99", unit="gallon")],
                store_name="Harris Teeter",
                confidence=0.8,
            ),
        ]

        export_results(extractions, str(tmp_path))

        json_file = tmp_path / "extraction_results.json"
        data = json.loads(json_file.read_text())
        assert len(data) == 2
        assert data[0]["confidence"] == 1.0
        assert data[1]["confidence"] == 0.8

    def test_export_creates_output_directory(self, tmp_path):
        """Create output directory if it doesn't exist."""
        output_dir = tmp_path / "nested" / "dir"
        assert not output_dir.exists()

        extraction = GroceryAdExtraction(
            page_number=1,
            items=[GroceryItem(product_name="Bananas", price="$0.59/lb")],
            confidence=1.0,
        )

        export_results([extraction], str(output_dir))

        assert output_dir.exists()
        assert (output_dir / "extraction_results.json").exists()
        assert (output_dir / "grocery_items.csv").exists()

    def test_export_with_confidence_calculation(self, tmp_path):
        """Calculate and print average confidence."""
        extractions = [
            GroceryAdExtraction(
                page_number=1,
                items=[GroceryItem(product_name="Bananas", price="$0.59/lb")],
                confidence=1.0,
            ),
            GroceryAdExtraction(
                page_number=2,
                items=[GroceryItem(product_name="Milk", price="$3.99")],
                confidence=0.8,
            ),
        ]

        # Capture print output
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            export_results(extractions, str(tmp_path))

        output = f.getvalue()
        assert "2 items from 2 page(s)" in output
        assert "Average confidence: 0.90" in output

    def test_export_empty_extractions(self, tmp_path):
        """Handle empty extractions list."""
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            export_results([], str(tmp_path))

        output = f.getvalue()
        # Should not print average confidence for empty list
        assert "0 items from 0 page(s)" in output

    def test_export_with_extended_fields(self, tmp_path):
        """Export items with all extended fields."""
        extraction = GroceryAdExtraction(
            page_number=1,
            items=[
                GroceryItem(
                    product_name="Ground Beef",
                    price="$4.99/lb",
                    unit="lb",
                    original_price="$5.99/lb",
                    discount_percent=17,
                    discount_description="Weekly Sale",
                )
            ],
            store_name="Kroger",
            page_date="Valid 1/15-1/21",
            confidence=0.95,
            extraction_notes="Clear image quality",
        )

        export_results([extraction], str(tmp_path))

        json_file = tmp_path / "extraction_results.json"
        data = json.loads(json_file.read_text())
        assert data[0]["page_date"] == "Valid 1/15-1/21"
        assert data[0]["extraction_notes"] == "Clear image quality"
        assert data[0]["items"][0]["original_price"] == "$5.99/lb"
        assert data[0]["items"][0]["discount_percent"] == 17
