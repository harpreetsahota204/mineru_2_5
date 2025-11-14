import logging
import sys
import io
import warnings
from contextlib import contextmanager
from typing import Union, List, Dict

from PIL import Image
import numpy as np
import torch

import fiftyone as fo
from fiftyone import Model
from fiftyone.core.models import SupportsGetItem
from fiftyone.utils.torch import GetItem

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from transformers.utils import is_flash_attn_2_available
from mineru_vl_utils import MinerUClient

logger = logging.getLogger(__name__)


@contextmanager
def suppress_output():
    """Suppress stdout, stderr, warnings, and transformers logging."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    
    # Suppress transformers logging
    transformers_logger = logging.getLogger("transformers")
    old_transformers_level = transformers_logger.level
    transformers_logger.setLevel(logging.ERROR)
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        transformers_logger.setLevel(old_transformers_level)


# Operation modes that determine extraction method and output format
OPERATIONS = {
    "layout_detection": {
        "method": "layout_detect",
        "return_type": "detections",
        "description": "FAST: Layout detection with bounding boxes only (1 inference pass)"
    },
    "ocr_detection": {
        "method": "two_step_extract",
        "return_type": "detections",
        "description": "SLOW: Full extraction with bounding boxes and OCR content (N+1 passes)"
    },
    "ocr": {
        "method": "content_extract",
        "return_type": "text",
        "description": "FAST: Plain text OCR extraction (1 inference pass)"
    }
}


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MinerUGetItem(GetItem):
    """GetItem transform for batching MinerU inference.
    
    Loads images as PIL Images for processing by MinerU.
    """
    
    @property
    def required_keys(self):
        """Fields required from each sample."""
        return ["filepath"]
    
    def __call__(self, sample_dict):
        """Load and return PIL Image.
        
        Args:
            sample_dict: Dictionary with 'filepath' key
            
        Returns:
            PIL.Image: RGB image loaded from filepath
        """
        filepath = sample_dict["filepath"]
        image = Image.open(filepath).convert("RGB")
        return image


class MinerU(Model, SupportsGetItem):
    """FiftyOne model for MinerU 2.5 document extraction with batching support.
    
    Supports three operation modes:
    - layout_detection: FAST - Layout detection with bounding boxes only (1 inference pass)
    - ocr_detection: SLOW - Full extraction with bounding boxes and OCR content (N+1 passes)
    - ocr: FAST - Plain text OCR extraction (1 inference pass)
    
    Args:
        model_path: HuggingFace model ID or local path (default: "opendatalab/MinerU2.5-2509-1.2B")
        operation: Task type - "layout_detection", "ocr_detection", "ocr" (default: "layout_detection")
        batch_size: Batch size for inference (default: 8)
    """
    
    def __init__(
        self,
        model_path: str = "opendatalab/MinerU2.5-2509-1.2B",
        operation: str = "layout_detection",
        batch_size: int = 8,
        **kwargs
    ):
        SupportsGetItem.__init__(self)
        self.model_path = model_path
        self._operation = operation
        self.batch_size = batch_size
        self._preprocess = False  # Preprocessing happens in GetItem
        
        # Validate operation
        if operation not in OPERATIONS:
            raise ValueError(
                f"operation must be one of {list(OPERATIONS.keys())}, "
                f"got '{operation}'"
            )
        
        # Device setup
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        
        # Load processor and model
        logger.info(f"Loading MinerU 2.5 from {model_path}")

        model_kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }

        if self.device == "cuda" and torch.cuda.is_available():            
            # Enable flash attention if available
            if is_flash_attn_2_available():
                model_kwargs["attn_implementation"] = "flash_attention_2"
        
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=True
        )
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            **model_kwargs
        )
        self.model = self.model.to(self.device).eval()
        
        # Initialize MinerU client with batch support
        self.client = MinerUClient(
            backend="transformers",
            model=self.model,
            processor=self.processor,
            batch_size=batch_size,
        )
        
        logger.info(f"MinerU 2.5 model loaded successfully (batch_size={batch_size})")
    
    @property
    def media_type(self):
        """The media type processed by this model."""
        return "image"
    
    @property
    def preprocess(self):
        """Whether preprocessing should be applied.
        
        For SupportsGetItem models, preprocessing is handled by the GetItem
        transform, so this should be False when using the DataLoader path.
        """
        return self._preprocess
    
    @preprocess.setter
    def preprocess(self, value):
        """Set preprocessing flag."""
        self._preprocess = value
    
    @property
    def has_collate_fn(self):
        """Whether this model provides a custom collate function.
        
        Returns True since we need custom collation for variable-size images.
        """
        return True
    
    @property
    def collate_fn(self):
        """Custom collate function for the DataLoader.
        
        Returns batches as lists of PIL Images without stacking,
        since MinerU handles variable-size images.
        """
        @staticmethod
        def identity_collate(batch):
            """Return batch as-is (list of PIL Images)."""
            return batch
        
        return identity_collate
    
    @property
    def ragged_batches(self):
        """Whether this model supports batches with varying sizes.
        
        Returns False to enable batching with batch_size > 1.
        MinerU handles variable-size PIL Images internally.
        """
        return False
    
    @property
    def transforms(self):
        """The preprocessing transforms applied to inputs.
        
        For SupportsGetItem models, preprocessing happens in the GetItem
        transform, so this returns None.
        """
        return None
    
    @property
    def operation(self):
        """Current operation mode."""
        return self._operation
    
    @operation.setter
    def operation(self, value):
        """Change operation mode at runtime."""
        if value not in OPERATIONS:
            raise ValueError(
                f"operation must be one of {list(OPERATIONS.keys())}, "
                f"got '{value}'"
            )
        self._operation = value
        logger.info(f"Operation changed to: {value}")
    
    def get_item(self):
        """Return the GetItem transform for batching support.
        
        Returns:
            MinerUGetItem: GetItem instance for loading images
        """
        return MinerUGetItem()
    
    def build_get_item(self, field_mapping=None):
        """Build the GetItem transform for batching.
        
        Args:
            field_mapping: Optional field mapping dict
            
        Returns:
            MinerUGetItem: GetItem instance for loading images
        """
        return MinerUGetItem(field_mapping=field_mapping)
    
    def _get_return_type(self):
        """Determine return type based on operation.
        
        Returns:
            str: "detections" or "text"
        """
        return OPERATIONS[self._operation]["return_type"]
    
    def _to_detections(self, blocks: List) -> fo.Detections:
        """Convert MinerU blocks to FiftyOne Detections.
        
        Args:
            blocks: List of ContentBlock objects or dicts with 'type', 'bbox', 'angle', 'content'
                   bbox format: [x1, y1, x2, y2] in normalized coordinates [0, 1]
        
        Returns:
            fo.Detections: FiftyOne Detections object with all parsed blocks
        """
        detections = []
        
        for block in blocks:
            # Handle both ContentBlock objects and dicts
            if hasattr(block, 'bbox'):
                # ContentBlock object (from layout_detect or two_step_extract)
                x1, y1, x2, y2 = block.bbox
                label = block.type
                angle = block.angle if block.angle is not None else 0
                text = block.content if block.content is not None else ''
            else:
                # Dict format (fallback)
                x1, y1, x2, y2 = block['bbox']
                label = block['type']
                angle = block.get('angle', 0)
                text = block.get('content', '')
            
            # Convert to FiftyOne format: [x, y, width, height]
            width = x2 - x1
            height = y2 - y1
            
            detection = fo.Detection(
                label=label,
                bounding_box=[x1, y1, width, height],
                angle=angle,
                text=text
            )
            
            detections.append(detection)
        
        return fo.Detections(detections=detections)
    
    def predict_all(self, images, preprocess=None):
        """Batch prediction for multiple images.
        
        This method enables efficient batching when using dataset.apply_model().
        
        Args:
            images: List of PIL Images (from GetItem) or numpy arrays
            preprocess: If True, convert numpy to PIL. If None, uses self.preprocess.
        
        Returns:
            List of predictions based on operation:
            - List[fo.Detections] for "layout_detection" or "ocr_detection"
            - List[str] for "ocr"
        """
        # Use instance preprocess flag if not specified
        if preprocess is None:
            preprocess = self._preprocess
        
        # Preprocess if needed (convert numpy to PIL)
        if preprocess:
            pil_images = []
            for img in images:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                elif not isinstance(img, Image.Image):
                    raise ValueError(f"Expected PIL Image or numpy array, got {type(img)}")
                pil_images.append(img)
            images = pil_images
        else:
            # Images should already be PIL Images from GetItem
            # But ensure they are
            if images and not isinstance(images[0], Image.Image):
                raise ValueError(
                    f"When preprocess=False, images must be PIL Images. "
                    f"Got {type(images[0]) if images else 'empty list'}"
                )
        
        # Get the batch method name based on operation
        method_name = OPERATIONS[self._operation]["method"]
        
        # Call the appropriate batch method
        with suppress_output():
            if method_name == "layout_detect":
                # Fast: Only layout detection (1 pass per image)
                blocks_list = self.client.batch_layout_detect(images)
                
            elif method_name == "two_step_extract":
                # Slow: Full extraction (N+1 passes per image)
                blocks_list = self.client.batch_two_step_extract(images)
                
            elif method_name == "content_extract":
                # Fast: Content extraction only (1 pass per image)
                results = self.client.batch_content_extract(images)
                return results
            else:
                raise ValueError(f"Unknown method: {method_name}")
        
        # Convert to detections if needed
        if self._get_return_type() == "detections":
            return [self._to_detections(blocks) for blocks in blocks_list]
        
        return blocks_list
    
    def _predict(self, image: Image.Image) -> Union[fo.Detections, str]:
        """Process image through MinerU (single image).
        
        Args:
            image: PIL Image to process
        
        Returns:
            - fo.Detections if operation="layout_detection" or "ocr_detection"
            - str if operation="ocr"
        """
        # Get the extraction method based on operation
        method_name = OPERATIONS[self._operation]["method"]
        method = getattr(self.client, method_name)
        
        # Run inference with suppressed output
        with suppress_output():
            result = method(image)
        
        # Parse output based on return type
        if self._get_return_type() == "detections":
            return self._to_detections(result)
        
        return result
    
    def predict(self, image, sample=None):
        """Process a single image with MinerU 2.5.
        
        For batch processing, use predict_all() or dataset.apply_model() which will
        automatically use batching via the GetItem interface.
        
        Args:
            image: PIL Image or numpy array to process
            sample: FiftyOne sample (optional, for compatibility)
        
        Returns:
            Model predictions in the appropriate format for the current operation:
            - fo.Detections for "layout_detection" or "ocr_detection"
            - str for "ocr"
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self._predict(image)
