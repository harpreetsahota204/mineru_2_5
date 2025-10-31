import logging
from typing import Union, List, Dict

from PIL import Image
import numpy as np
import torch

import fiftyone as fo
from fiftyone import Model, SamplesMixin

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from transformers.utils import is_flash_attn_2_available
from mineru_vl_utils import MinerUClient

logger = logging.getLogger(__name__)


# Operation modes that determine extraction method and output format
OPERATIONS = {
    "ocr_detection": {
        "method": "ocr_detection_extract",
        "return_type": "detections",
        "description": "Structured document extraction with bounding boxes"
    },
    "ocr": {
        "method": "content_extract",
        "return_type": "text",
        "description": "Plain text OCR extraction"
    }
}


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MinerU(Model, SamplesMixin):
    """FiftyOne model for MinerU 2.5 document extraction.
    
    Supports two operation modes:
    - ocr_detection: Structured extraction with bounding boxes (returns fo.Detections)
    - ocr: Plain text OCR extraction (returns str)
    
    Automatically selects optimal dtype based on hardware:
    - bfloat16 for CUDA devices with compute capability 8.0+ (Ampere and newer)
    - float16 for older CUDA devices
    - float32 for CPU/MPS devices
    
    Args:
        model_path: HuggingFace model ID or local path (default: "opendatalab/MinerU2.5-2509-1.2B")
        operation: Task type - "ocr_detection", "ocr" (default: "ocr_detection")
    """
    
    def __init__(
        self,
        model_path: str = "opendatalab/MinerU2.5-2509-1.2B",
        operation: str = "ocr_detection",
        **kwargs
    ):
        SamplesMixin.__init__(self)
        self.model_path = model_path
        self._operation = operation
        
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
            "dtype": self.dtype,
            "device_map": self.device,
        }
        model_kwargs.update(kwargs)

        # Device setup
        self.device = get_device()
        logger.info(f"Using device: {self.device}")

        # Load processor and model
        logger.info(f"Loading MinerU 2.5 from {model_path}")

        model_kwargs = {
            "device_map": self.device,
        }

        if self.device == "cuda" and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            
            # Enable flash attention if available
            if is_flash_attn_2_available():
                model_kwargs["attn_implementation"] = "flash_attention_2"
            
            # Enable bfloat16 on Ampere+ GPUs (compute capability 8.0+)
            if capability[0] >= 8:
                model_kwargs["dtype"] = torch.bfloat16
        
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=True
        )
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            **model_kwargs
        )
        self.model = self.model.eval()
        
        # Initialize MinerU client
        self.client = MinerUClient(
            backend="transformers",
            model=self.model,
            processor=self.processor
        )
        
        logger.info("MinerU 2.5 model loaded successfully")
    
    @property
    def media_type(self):
        """The media type processed by this model."""
        return "image"
    
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
    
    def _get_return_type(self):
        """Determine return type based on operation.
        
        Returns:
            str: "detections" or "text"
        """
        return OPERATIONS[self._operation]["return_type"]
    
    def _to_detections(self, blocks: List[Dict]) -> fo.Detections:
        """Convert MinerU blocks to FiftyOne Detections.
        
        Args:
            blocks: List of dicts with 'type', 'bbox', 'angle', 'text'
                   bbox format: [x1, y1, x2, y2] in normalized coordinates [0, 1]
        
        Returns:
            fo.Detections: FiftyOne Detections object with all parsed blocks
        """
        detections = []
        
        for block in blocks:
            # bbox is already [x1, y1, x2, y2] in normalized coords
            x1, y1, x2, y2 = block['bbox']
            
            # Convert to FiftyOne format: [x, y, width, height]
            width = x2 - x1
            height = y2 - y1
            
            detection = fo.Detection(
                label=block['type'],
                bounding_box=[x1, y1, width, height],
                angle=block.get('angle', 0),
                text=block.get('content', '')
            )
            
            detections.append(detection)
        
        return fo.Detections(detections=detections)
    
    def _predict(self, image: Image.Image) -> Union[fo.Detections, str]:
        """Process image through MinerU.
        
        Args:
            image: PIL Image to process
        
        Returns:
            - fo.Detections if operation="ocr_detection" (with bounding boxes)
            - str if operation="content" (plain text)
        """
        # Get the extraction method based on operation
        method_name = OPERATIONS[self._operation]["method"]
        method = getattr(self.client, method_name)
        
        # Run inference
        result = method(image)
        
        # Parse output based on return type
        if self._get_return_type() == "detections":
            return self._to_detections(result)
        
        return result
    
    def predict(self, image, sample=None):
        """Process an image with MinerU 2.5.
        
        Args:
            image: PIL Image or numpy array to process
            sample: FiftyOne sample (not required for MinerU, included for compatibility)
        
        Returns:
            Model predictions in the appropriate format for the current operation:
            - fo.Detections for "ocr_detection" operation
            - str for "content" operation
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self._predict(image)
