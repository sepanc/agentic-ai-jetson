from langchain_core.utils.utils import raise_for_status_with_text
from pydantic import BaseModel, Field, field_validator
from typing import List, Literal
import json

class BoundingBox(BaseModel):
    """Bounding box coordinates for detected object."""
    x: int = Field(ge=0, description="X coordinate (top-left)")
    y: int = Field(ge=0, description="Y coordinate (top-left)")
    width: int = Field(gt=0, description="Width in pixels")
    height: int = Field(gt=0, description="Height in pixels")

class Detection(BaseModel):
    """Single object detection with validation."""
    object: str = Field(min_length=1, max_length=50, description="Object label")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    bounding_box: BoundingBox
    action: Literal["track", "ignore", "alert"] = Field(description="Recommended action")
    
    @field_validator('object')
    @classmethod
    def validate_object_label(cls, v: str) -> str:
        """Ensure object label is clean."""
        cleaned = v.strip().lower()
        if not cleaned:
            raise ValueError("Object label cannot be empty")
        return cleaned

class VisionResponse(BaseModel):
    """Complete vision detection response with multiple objects."""
    detections: List[Detection]
    timestamp: str
    frame_id: int = Field(ge=0)

from langchain_ollama import ChatOllama
from pydantic import ValidationError
from datetime import datetime
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StructuredVisionAgent:
    """Vision agent with structured output and retry logic."""
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        """Initialize agent with Ollama LLM."""
        self.llm = ChatOllama(
            model="llama3.2:3b",
            base_url=ollama_base_url,
            temperature=0.1,  # Low temperature for more deterministic output
            num_predict=512,
            format='json'
        )
    
    def detect_objects(self, scene_description: str, max_retries: int = 3) -> Optional[VisionResponse]:
        """
        Detect objects in scene with structured output and retry logic.
        
        Args:
            scene_description: Text description of what camera sees
            max_retries: Maximum retry attempts on validation failure
            
        Returns:
            VisionResponse with validated detections, or None if all retries fail
        """
        # Build initial prompt
        prompt = self._build_detection_prompt(scene_description)
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries}")
                logger.info(f"Prompt: {prompt}")

                # Get LLM response
                response = self.llm.invoke(prompt)
                raw_text = response.content
                
                logger.info(f"LLM returned: {raw_text}...")
                
                raw_json = self._extract_json(raw_text)
                logger.info(f"Raw JSON: {raw_json}")
                
                # Validate with Pydantic
                vision_response = VisionResponse.model_validate_json(raw_json)
                
                logger.info(f"✅ Validation successful on attempt {attempt + 1}")
                return vision_response
                
            except ValidationError as e:
                logger.warning(f"❌ Validation failed on attempt {attempt + 1}: {e}")
                
                if attempt < max_retries - 1:
                    # Retry with error feedback
                    prompt = self._build_retry_prompt(scene_description, raw_json, str(e))
                else:
                    logger.error(f"All {max_retries} attempts failed")
                    return None
            
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return None
        
        return None
    
    def _extract_json(self, text: str) -> str:
        """Extract pure JSON from response that may have markdown/preamble/postamble."""
        cleaned = text.replace('```json', '').replace('```', '').strip()
        # Find first { and last }
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1:   
            return cleaned[start:end+1].strip() # strip to remove whitespace
        else: # Fallback: return cleaned text
            return cleaned.strip()

    def _build_detection_prompt(self, scene_description: str) -> str:
        """Build initial detection prompt with JSON schema."""
        return f"""You are a vision detection system. Analyze the scene and return detections in JSON format.

Scene: {scene_description}

Return ONLY valid JSON matching this exact structure (no markdown, no explanations):
{{
    "detections": [
        {{
            "object": "person",
            "confidence": 0.95,
            "bounding_box": {{
                "x": 100,
                "y": 150,
                "width": 80,
                "height": 200
            }},
            "action": "track"
        }}
    ],
    "timestamp": "{datetime.now().isoformat()}",
    "frame_id": 0
}}

Rules:
- object: string (e.g., "person", "car", "dog")
- confidence: float between 0.0 and 1.0
- bounding_box: x, y, width, height as positive integers
- action: must be one of: "track", "ignore", or "alert"
- Return empty detections array if nothing detected

Return JSON now:"""

    def _build_retry_prompt(self, scene_description: str, failed_json: str, error_msg: str) -> str:
        """Build retry prompt with error feedback."""
        return f"""You are a vision detection system. Your previous response had validation errors.

Scene: {scene_description}

Your previous response:
{failed_json}

Validation errors:
{error_msg}

Fix the errors and return ONLY valid JSON matching this exact structure:
{{
    "detections": [
        {{
            "object": "person",
            "confidence": 0.95,
            "bounding_box": {{
                "x": 100,
                "y": 150,
                "width": 80,
                "height": 200
            }},
            "action": "track"
        }}
    ],
    "timestamp": "{datetime.now().isoformat()}",
    "frame_id": 0
}}

Common fixes:
- confidence must be float 0.0-1.0 (not "high" or "very confident")
- action must be exactly "track", "ignore", or "alert"
- bounding_box coordinates must be positive integers
- object names must be lowercase strings

Return corrected JSON now:"""

# Add this at the bottom of structured_vision.py
if __name__ == "__main__":
    # Test with simulated scene
    agent = StructuredVisionAgent(ollama_base_url="http://localhost:11434")
    
    scene = "Camera sees a person standing near a car in a parking lot"
    
    print("🔍 Testing vision detection with retry logic...\n")
    result = agent.detect_objects(scene)
    
    if result:
        print("\n✅ SUCCESS - Structured output:")
        print(f"Frame ID: {result.frame_id}")
        print(f"Timestamp: {result.timestamp}")
        print(f"Detections: {len(result.detections)}")
        for det in result.detections:
            print(f"  - {det.object}: {det.confidence:.2f} [{det.action}]")
    else:
        print("\n❌ FAILED - Could not get valid response")