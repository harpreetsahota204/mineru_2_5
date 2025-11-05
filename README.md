# MinerU 2.5 for FiftyOne

![image](mineru2_5.gif)

A FiftyOne zoo model integration for [MinerU2.5](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B), a 1.2B-parameter vision-language model for efficient high-resolution document parsing.

## About MinerU 2.5

MinerU2.5 achieves state-of-the-art document parsing accuracy with a two-stage strategy:
- **Global layout analysis** on downsampled images
- **Fine-grained content recognition** on native-resolution crops

Key capabilities:
- Comprehensive layout analysis (headers, footers, lists, code blocks)
- Complex mathematical formula parsing (including mixed Chinese-English)
- Robust table parsing (handles rotated, borderless, and partial-border tables)

For more details, see the [model card](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B) and [technical report](https://arxiv.org/abs/2509.22186).

## Installation

```bash
pip install fiftyone
pip install "mineru-vl-utils[transformers]"
```

## Usage in FiftyOne

### Register the Model Source

```python
import fiftyone.zoo as foz

foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/mineru_2_5",
    overwrite=True
)
```

### Load the Model

```python
# Load with default settings
model = foz.load_zoo_model("opendatalab/MinerU2.5-2509-1.2B")
```

### OCR Detection (Structured Output with Bounding Boxes)

Extract document structure with bounding boxes for each element:

```python
import fiftyone as fo

# Load your dataset
dataset = fo.load_dataset("your-dataset")

# Apply model for structured extraction
model.operation = "ocr_detection"
dataset.apply_model(model, label_field="text_detections")
```

This returns `fo.Detections` with:
- **label**: Element type (`title`, `text`, `table`, `figure`, etc.)
- **bounding_box**: Normalized coordinates `[x, y, width, height]`
- **text**: Extracted text content
- **angle**: Rotation angle (if applicable)

### OCR (Plain Text Extraction)

Extract all text content as a single string:

```python
model.operation = "ocr"
dataset.apply_model(model, label_field="text_extraction")
```

This returns plain text strings with the OCR'd content.

## Complete Example

```python
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone.utils.huggingface import load_from_hub

# Load dataset
dataset = load_from_hub(
    "harpreetsahota/NutriGreen",
    name="NutriGreen_MinerU25",
    max_samples=5
)

# Register and load model
foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/mineru_2_5",
    overwrite=True
)
model = foz.load_zoo_model("opendatalab/MinerU2.5-2509-1.2B")

# Apply structured extraction
model.operation = "ocr_detection"
dataset.apply_model(model, label_field="text_detections")

# Apply text extraction
model.operation = "ocr"
dataset.apply_model(model, label_field="text_extraction")

# Launch the app
session = fo.launch_app(dataset)
```

## Model Configuration

### Operations

- `ocr_detection` (default): Structured extraction with bounding boxes → returns `fo.Detections`
- `ocr`: Plain text OCR → returns `str`

### Advanced Options

Pass additional parameters via `kwargs`:

```python
model = foz.load_zoo_model(
    "opendatalab/MinerU2.5-2509-1.2B",
    operation="ocr_detection",
    torch_dtype=torch.float16  # Override dtype
)
```

The model automatically:
- Selects the best available device (CUDA/MPS/CPU)
- Uses `dtype="auto"` for optimal precision
- Enables Flash Attention 2 when available on CUDA devices

## Requirements

- Python 3.8+
- FiftyOne
- PyTorch
- transformers >= 4.56.0 (recommended)
- mineru-vl-utils

## Citation

```bibtex
@misc{niu2025mineru25decoupledvisionlanguagemodel,
    title={MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing}, 
    author={Junbo Niu and Zheng Liu and Zhuangcheng Gu and Bin Wang and Linke Ouyang and Zhiyuan Zhao and Tao Chu and Tianyao He and Fan Wu and Qintong Zhang and Zhenjiang Jin and others},
    year={2025},
    eprint={2509.22186},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2509.22186}
}
```

## License

AGPL-3.0 (as per the original MinerU2.5 model)
