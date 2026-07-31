<!-- @dsCard group="Examples" -->

# Grocery Ad PDF Extractor

Extract products and prices from grocery store advertisement PDFs using Mellea and Granite vision models. Two implementations are provided: a straightforward basic extractor and an advanced version with validation and retry logic.

## Overview

These examples demonstrate how to use Mellea for multi-page PDF processing with vision models:

- **`grocery_ad_extractor.py`** — Basic implementation for straightforward PDF extraction
- **`grocery_ad_extractor_advanced.py`** — Enhanced version with requirements validation, retry logic, and confidence scoring

Both examples extract structured data (product name, price, unit, discounts) from PDF images and export results in multiple formats (CSV, JSON, Markdown).

## Prerequisites

- **Ollama** running with a vision model:
  ```bash
  ollama pull hf.co/ibm-granite/granite-vision-4.1-4b-GGUF:Q4_K_M
  ```
- **Python dependencies**:
  ```bash
  uv pip install pydantic pdf2image pillow
  ```

## Basic Extractor (`grocery_ad_extractor.py`)

Simple, single-session approach for extracting grocery items from PDFs.

### Features

- Single-pass extraction from each PDF page
- Structured output with Pydantic models
- Export to CSV and Markdown
- Debug page images saved for inspection

### Usage

```bash
python grocery_ad_extractor.py path/to/grocery_ad.pdf
python grocery_ad_extractor.py HarrisTeeter.pdf ollama --no-debug
```

### Example Output

**CSV Export** (`HarrisTeeter_output.csv`):
```
Page,Store,Product,Price,Unit,Discount
1,Harris Teeter,Organic Bananas,$0.59/lb,lb,
1,Harris Teeter,Ground Beef,$5.99/lb,lb,Buy 2 Get 1 Free
```

**Markdown Export** (`HarrisTeeter_output.md`):
```markdown
## Page 1

**Store:** Harris Teeter

| Product | Price | Unit | Discount |
|---------|-------|------|----------|
| Organic Bananas | $0.59/lb | lb |  |
| Ground Beef | $5.99/lb | lb | Buy 2 Get 1 Free |
```

### Data Model

```python
class GroceryItem(BaseModel):
    product_name: str
    price: str
    unit: str | None
    discount: str | None

class GroceryAdExtraction(BaseModel):
    items: list[GroceryItem]
    store_name: str | None
    notes: str | None
```

### Key Implementation Details

- **Per-page session isolation**: Creates a fresh Mellea session for each page to avoid context pollution
- **Structured output**: Uses `format=GroceryAdExtraction` for JSON schema validation
- **Error handling**: Graceful per-page error handling with progress reporting

```python
with start_session(model_id=model_id, model_options=model_opts, ctx=ChatContext()) as m:
    img_block = ImageBlock.from_pil_image(image)
    result = m.instruct(prompt, images=[img_block], format=GroceryAdExtraction)
    extracted = GroceryAdExtraction.model_validate_json(str(result))
```

## Advanced Extractor (`grocery_ad_extractor_advanced.py`)

Production-ready extractor with validation, retry logic, and confidence scoring.

### Features

- **Requirements validation**: Enforces constraints on extracted data
  - Prices must be in currency format
  - Product names must be non-empty
  - At least one product per page
- **Retry logic**: Automatic retries on extraction failures (default: 2 attempts)
- **Confidence scoring**: Lower scores (0.8) for retried extractions
- **Multiple export formats**: JSON, CSV with extended columns
- **Extensible design**: Class-based interface for custom initialization

### Usage

```bash
python grocery_ad_extractor_advanced.py path/to/grocery_ad.pdf
python grocery_ad_extractor_advanced.py weekly_ad.pdf --no-debug
```

### Example Output

**JSON Export** (`extraction_results.json`):
```json
[
  {
    "page_number": 1,
    "store_name": "Harris Teeter",
    "items": [
      {
        "product_name": "Organic Bananas",
        "price": "$0.59/lb",
        "unit": "lb",
        "original_price": null,
        "discount_percent": null,
        "discount_description": null
      }
    ],
    "page_date": "Valid through Sunday",
    "confidence": 1.0,
    "extraction_notes": null
  }
]
```

**CSV Export** (`grocery_items.csv`):
```
Page,Store,Product,Price,Unit,Original Price,Discount %,Promotion,Confidence
1,Harris Teeter,Organic Bananas,$0.59/lb,lb,,,,1.0
```

### Data Model

```python
class GroceryItem(BaseModel):
    product_name: str
    price: str
    unit: str | None
    original_price: str | None
    discount_percent: int | None
    discount_description: str | None

class GroceryAdExtraction(BaseModel):
    page_number: int
    items: list[GroceryItem]
    store_name: str | None
    page_date: str | None
    confidence: float  # 0-1
    extraction_notes: str | None
```

### Key Implementation Details

- **Context manager pattern**: Reusable `GroceryAdExtractorAdvanced` class
- **Requirements validation**: 
  ```python
  requirements = [
      Requirement("All prices must be in currency format (e.g., $X.XX)"),
      Requirement("Product names must be non-empty"),
      Requirement("At least one product must be extracted"),
  ] if validate else []
  
  result = session.instruct(prompt, images=[img_block], format=GroceryAdExtraction, requirements=requirements)
  ```
- **Retry with confidence**:
  ```python
  extraction.confidence = 1.0 if attempt == 1 else 0.8
  ```

## Comparing the Two Approaches

| Feature | Basic | Advanced |
|---------|-------|----------|
| **Simplicity** | ✓ Straightforward | Complex setup |
| **Validation** | None | Enforced via requirements |
| **Retry logic** | None | Auto-retry on failure |
| **Confidence scores** | N/A | Included |
| **Export formats** | CSV, Markdown | JSON, CSV |
| **Error recovery** | Per-page skip | Per-page with retry |
| **Reusability** | Function-based | Class-based |

Choose **basic** for simple one-off extractions or when you want minimal dependencies. Choose **advanced** for production pipelines where data quality and completeness are critical.

## Vision Model Integration

Both examples use Mellea's vision model support:

```python
from mellea import start_session
from mellea.core import ImageBlock

# Initialize with a vision model
model_opts: dict[str, int] = {ModelOption.CONTEXT_WINDOW: 16384}
with start_session(model_id=model_id, model_options=model_opts, ctx=ChatContext()) as m:
    # Convert PIL Image to ImageBlock
    img_block = ImageBlock.from_pil_image(image)
    
    # Extract with structured output
    result = m.instruct(
        prompt,
        images=[img_block],
        format=GroceryAdExtraction,
        requirements=requirements,  # Advanced only
    )
```

**Supported models:**
- `hf.co/ibm-granite/granite-vision-4.1-4b-GGUF:Q4_K_M` (Ollama, local)
- `hf.co/ibm-granite/granite-vision-4.1-8b-GGUF:Q4_K_M` (Ollama, local)
- `gpt-4-vision` (OpenAI backend)

## Common Patterns

### Per-Page Session Isolation

Both examples create a fresh session per page to avoid context accumulation:

```python
for page_num, image in enumerate(images, 1):
    with start_session(model_id=model_id, model_options=model_opts, ctx=ChatContext()) as m:
        # Fresh context for each page
```

### Structured Output with Pydantic

Pydantic models are passed to `format=` for JSON schema validation:

```python
result = m.instruct(prompt, images=[img_block], format=GroceryAdExtraction)
extracted = GroceryAdExtraction.model_validate_json(str(result))
```

The LLM is prompted to return valid JSON matching the Pydantic schema.

### Error Handling

Both examples use try/except with per-page granularity so failures don't stop the entire pipeline:

```python
for page_num, image in enumerate(images, 1):
    try:
        extraction = self.extract_page(image, page_num, validate=validate)
        if extraction:
            results.append(extraction)
    except Exception as e:
        print(f"✗ Error processing page {page_num}: {e}")
        continue  # Continue to next page
```

## Extending the Examples

### Add a New Field to GroceryItem

1. Update the Pydantic model:
   ```python
   class GroceryItem(BaseModel):
       product_name: str
       price: str
       unit: str | None
       sale_dates: str | None  # NEW
   ```

2. Update the prompt to request the new field:
   ```python
   prompt = (
       "Extract: product_name, price, unit, sale_dates (e.g., 'Valid 1/15-1/21').\n"
       "Return structured JSON."
   )
   ```

3. Update export functions to include the new field in CSV/JSON.

### Switch Vision Models

Change the `model_id` when initializing:

```python
# Use a different Granite model
model_id = "hf.co/ibm-granite/granite-vision-4.1-8b-GGUF:Q4_K_M"

# Or switch to OpenAI
model_id = "gpt-4-vision"
```

### Add Custom Requirements

In the advanced extractor, add new `Requirement` objects:

```python
requirements = [
    Requirement("All prices must be in currency format (e.g., $X.XX)"),
    Requirement("Product names must be non-empty"),
    Requirement("Discount percentages must be between 0 and 100"),  # NEW
]
```

## Troubleshooting

### "Ollama refused" or connection error

Ensure Ollama is running:
```bash
ollama serve
```

### "ImageBlock validation failed"

The vision model may have rejected the image format. Verify the image is a valid PNG/JPG:
```python
img_block = ImageBlock.from_pil_image(image)  # Always use this, not raw base64
```

### Extraction returns empty or None

- Check the PDF is readable (not scanned as text-only)
- Verify the prompt clearly describes the task
- Try increasing `CONTEXT_WINDOW` in model options

### Division by zero in advanced extractor

The code already guards against this:
```python
if extractions:
    print(f"Average confidence: {avg_confidence:.2f}")
```

But verify at least one page extracted successfully.

## References

- [Mellea Vision Models Documentation](../docs/vision.md)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [pdf2image Documentation](https://pypi.org/project/pdf2image/)
