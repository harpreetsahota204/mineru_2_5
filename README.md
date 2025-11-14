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

### Optional: Install Caption Viewer Plugin

For the best experience viewing extracted text, we recommend installing the [Caption Viewer plugin](https://github.com/harpreetsahota204/caption_viewer), which provides intelligent formatting for OCR outputs and text fields:

```bash
fiftyone plugins download https://github.com/harpreetsahota204/caption_viewer
```

This plugin automatically:
- Renders line breaks and escape sequences properly
- Converts HTML tables to markdown
- Pretty-prints JSON content
- Shows character counts

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
# Load with default settings (fast layout detection mode)
model = foz.load_zoo_model(
    "opendatalab/MinerU2.5-2509-1.2B",
    operation="layout_detection",  # Fast mode (default)
    batch_size=8  # Process 8 images at once
)
```

### Three Operation Modes

MinerU now supports three operation modes with different speed/accuracy tradeoffs:

#### 1. Layout Detection (FAST - Recommended) ⚡

Get bounding boxes for all document elements with **1 inference pass per image**:

```python
import fiftyone as fo

# Load your dataset
dataset = fo.load_dataset("your-dataset")

# Apply fast layout detection (default)
model.operation = "layout_detection"
dataset.apply_model(model, label_field="layout")
```

**Speed**: ~50ms per image (1.2B model on GPU)  
**Returns**: `fo.Detections` with bounding boxes and element types  
**Use case**: Quick document analysis, element counting, layout understanding

#### 2. OCR Detection (SLOW but Complete) 🐢

Full extraction with bounding boxes AND OCR content:

```python
model.operation = "ocr_detection"
dataset.apply_model(model, label_field="text_detections")
```

**Speed**: ~800ms per image (15 blocks × 50ms + overhead)  
**Returns**: `fo.Detections` with bounding boxes, types, AND extracted text  
**Use case**: Complete document extraction with precise element locations

#### 3. OCR (FAST - Text Only) ⚡

Extract all text content as a single string:

```python
model.operation = "ocr"
dataset.apply_model(model, label_field="text_extraction")
```

**Speed**: ~50ms per image  
**Returns**: Plain text string  
**Use case**: Full-text search, content indexing

### Detection Format

All detection modes return `fo.Detections` with:
- **label**: Element type (`text`, `title`, `table`, `equation`, `image`, etc.)
- **bounding_box**: Normalized coordinates `[x, y, width, height]`
- **text**: Extracted text content (empty for `layout_detection` mode)
- **angle**: Rotation angle (0, 90, 180, or 270)

## Complete Example

```python
import fiftyone as fo
import fiftyone.zoo as foz

# Register and load model
foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/mineru_2_5",
    overwrite=True
)

# Load model with batching support (NEW!)
model = foz.load_zoo_model(
    "opendatalab/MinerU2.5-2509-1.2B",
    operation="layout_detection",  # Fast mode (default)
    batch_size=8  # Process 8 images at once
)

# Load your dataset
dataset = fo.Dataset.from_dir(
    dataset_dir="/path/to/images",
    dataset_type=fo.types.ImageDirectory,
)

# Apply FAST layout detection (with batching!)
model.operation = "layout_detection"
dataset.apply_model(model, label_field="layout")

# Optionally: Apply full OCR to specific samples
# model.operation = "ocr_detection"
# dataset.match(F("layout.detections").length() > 5).apply_model(
#     model, label_field="detailed_ocr"
# )

# Launch the app
session = fo.launch_app(dataset)
```

## Performance Comparison

| Operation | Inference Passes | Speed (per image) | Use Case |
|-----------|-----------------|-------------------|----------|
| `layout_detection` | **1 pass** | ~50ms | Fast bounding boxes only ⚡ |
| `ocr` | **1 pass** | ~50ms | Fast text extraction ⚡ |
| `ocr_detection` | **N+1 passes** | ~800ms | Complete extraction 🐢 |

**Batching speedup**: With `batch_size=8`, you get an additional **3-5x speedup** from GPU parallelization!

## Model Configuration

### Operations

- `layout_detection` (default): Fast layout with bounding boxes → `fo.Detections` (1 pass)
- `ocr_detection`: Full extraction with content → `fo.Detections` (N+1 passes)
- `ocr`: Plain text only → `str` (1 pass)

### Batching (NEW!)

The model now supports efficient batching via FiftyOne's `SupportsGetItem` interface:

```python
# Load model with initial batch size
model = foz.load_zoo_model(
    "opendatalab/MinerU2.5-2509-1.2B",
    operation="layout_detection",
    batch_size=8,  # Internal MinerU batch size (default: 8)
)

# Batching happens automatically with apply_model()
dataset.apply_model(
    model, 
    label_field="results",
    batch_size=8,      # FiftyOne DataLoader batch size
    num_workers=4      # Parallel data loading workers
)
```

**Why set batch_size in two places?**
- `model.batch_size`: Controls MinerU's internal inference batching (how many images the VLM processes at once)
- `apply_model(..., batch_size=...)`: Controls FiftyOne's DataLoader batching (how many images are loaded in parallel)

For best performance, **set both to the same value**:

```python
BATCH_SIZE = 32

# Update model's internal batch size
model.batch_size = BATCH_SIZE
model.operation = "layout_detection"

# Apply with matching DataLoader batch size
dataset.apply_model(
    model, 
    label_field="layout",
    batch_size=BATCH_SIZE,  # Match model.batch_size
    num_workers=8           # 8 workers for parallel I/O
)
```

**Benefits**:
- **5-10x faster** than sequential processing
- 70-90% GPU utilization (vs 20-30% without batching)
- Parallel data loading with `num_workers`
- Optimal throughput when both batch sizes match

### Advanced Options

The model automatically:
- Selects the best available device (CUDA/MPS/CPU)
- Uses `torch.float16` on CUDA, `float32` on CPU/MPS
- Enables Flash Attention 2 when available on CUDA devices
- Optimizes model loading (no `device_map` overhead for small models)

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
