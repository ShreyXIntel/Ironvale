"""
UI Analyzer module for detecting UI elements in game screenshots.
"""
import logging
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger("UIAnalyzer")

class UIAnalyzer:
    """Analyzes game UI using Qwen 2.5 VL model."""
    
    def __init__(self, model_config: Dict[str, Any]):
        """Initialize the UI analyzer with model configuration.
        
        Args:
            model_config: Configuration for the Qwen model
        """
        self.model_config = model_config
        self.model = None
        self.processor = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the Qwen 2.5 VL model."""
        try:
            logger.info("Loading Qwen2.5-VL model...")
            
            # Convert string dtype to torch dtype
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16, 
                "bfloat16": torch.bfloat16
            }
            torch_dtype = dtype_map.get(self.model_config["torch_dtype"], torch.bfloat16)
            
            # Load model
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_config["model_path"],
                torch_dtype=torch_dtype,
                attn_implementation=self.model_config["attn_implementation"],
                device_map=self.model_config["device"],
            )
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_config["model_path"])
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def analyze_screenshot(self, screenshot_path: str, ui_prompt: str) -> Dict:
        """Analyze a screenshot and extract UI information.
        
        Args:
            screenshot_path: Path to the screenshot image
            ui_prompt: Prompt for the model to analyze the UI
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Analyzing screenshot: {screenshot_path}")
        
        try:
            # Prepare message for model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": screenshot_path},
                        {"type": "text", "text": ui_prompt}
                    ]
                }
            ]
            
            # Prepare inputs for model
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process images
            image_inputs, _ = self._process_vision_info(messages)
            
            # Create inputs for the model
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=None,  # Use None instead of empty list for videos
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model_config["device"])
            
            # Generate response
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=self.model_config["max_new_tokens"],
                temperature=self.model_config["temperature"],
                top_p=self.model_config["top_p"],
                repetition_penalty=self.model_config["repetition_penalty"]
            )
            
            # Process the response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # Parse the model output
            return self._parse_model_output(output_text)
        except Exception as e:
            logger.error(f"Error analyzing screenshot: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._create_default_analysis_result()
        finally:
            # Clean up resources
            self._cleanup_resources()
    
    def _process_vision_info(self, messages):
        """Process vision information from messages."""
        images = []
        
        for message in messages:
            content = message.get("content", [])
            if not isinstance(content, list):
                content = [content]
                
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    image_path = item.get("image")
                    if image_path:
                        try:
                            # Load and convert image to RGB
                            image = Image.open(image_path)
                            if image.mode != "RGB":
                                image = image.convert("RGB")
                            images.append(image)
                        except Exception as e:
                            logger.error(f"Error processing image {image_path}: {e}")
        
        return images, None  # Return None for videos as required by Qwen
    
    def _parse_model_output(self, output_text: str) -> Dict:
        """Parse the structured output from the model.
        
        Args:
            output_text: Raw text output from the model
            
        Returns:
            Structured dictionary with analysis results
        """
        result = self._create_default_analysis_result()
        result["raw_text"] = output_text
        
        # Parse the context
        if "CONTEXT:" in output_text:
            context_section = output_text.split("CONTEXT:")[1]
            context_line = context_section.split("\n")[0].strip()
            result["context"] = context_line
            logger.info(f"Detected context: {context_line}")
        
        # Parse UI elements
        if "DETECTED UI ELEMENTS:" in output_text:
            self._parse_ui_elements(output_text, result)
        
        # Parse navigation action
        if "NAVIGATION ACTION:" in output_text:
            self._parse_navigation_action(output_text, result)
        
        # Check for benchmark-related indicators
        self._check_benchmark_indicators(output_text, result)
        
        return result
    
    def _create_default_analysis_result(self) -> Dict:
        """Create a default analysis result structure."""
        return {
            "context": None,
            "ui_elements": [],
            "action": {
                "type": None,
                "coordinates": None,
                "confidence": 0.0,
                "reasoning": None
            },
            "is_benchmark_option": False,
            "is_benchmark_result": False,
            "raw_text": ""
        }
    
    def _parse_ui_elements(self, output_text: str, result: Dict) -> None:
        """Parse UI elements from the model output."""
        try:
            elements_section = output_text.split("DETECTED UI ELEMENTS:")[1]
            end_markers = ["NAVIGATION ACTION:", "REASONING:", "ACTION:"]
            end_pos = len(elements_section)
            
            for marker in end_markers:
                if marker in elements_section:
                    marker_pos = elements_section.find(marker)
                    if marker_pos > 0 and marker_pos < end_pos:
                        end_pos = marker_pos
            
            elements_section = elements_section[:end_pos].strip()
            element_lines = [line.strip() for line in elements_section.split("\n") if line.strip()]
            
            for line in element_lines:
                try:
                    if ":" in line and "[" in line and "]" in line:
                        # Extract element name
                        name_part = line.split(":")[0].strip()
                        element_name = name_part.lstrip("- •·*").strip()
                        
                        # Find coordinates inside brackets
                        coords_start = line.find("[")
                        coords_end = line.find("]", coords_start)
                        coords_part = line[coords_start:coords_end+1]
                        
                        # Parse coordinates
                        coords_str = coords_part.strip("[]")
                        if "," in coords_str:
                            coords_values = coords_str.split(",")
                        else:
                            coords_values = coords_str.split()
                        
                        # Convert to integers
                        coords = []
                        for val in coords_values:
                            clean_val = val.strip()
                            if clean_val.isdigit() or (clean_val.startswith("-") and clean_val[1:].isdigit()):
                                coords.append(int(clean_val))
                        
                        # Ensure we have 4 coordinates
                        if len(coords) == 4:  # x1, y1, x2, y2
                            # Extract description if available
                            description = ""
                            if "-" in line[coords_end:]:
                                desc_part = line[coords_end:].split("-", 1)[1].strip()
                                description = desc_part
                            
                            element = {
                                "name": element_name,
                                "coordinates": coords,
                                "description": description
                            }
                            result["ui_elements"].append(element)
                except Exception as e:
                    logger.warning(f"Failed to parse element line: {line}")
                    logger.warning(f"Error: {e}")
        except Exception as e:
            logger.warning(f"Failed to parse UI elements section: {e}")
    
    def _parse_navigation_action(self, output_text: str, result: Dict) -> None:
        """Parse navigation action from the model output."""
        # First try to find the main NAVIGATION ACTION section
        if "NAVIGATION ACTION:" in output_text:
            action_section = output_text.split("NAVIGATION ACTION:")[1].strip()
        else:
            # Fallback: look for action-related content anywhere in the text
            action_section = output_text
        
        action_lines = [line.strip() for line in action_section.split("\n") if line.strip()]
        
        # Track if we found any action information
        found_action = False
        
        # Process only the first complete action found
        for line in action_lines:
            try:
                if "ACTION_TYPE:" in line:
                    action_type = line.split("ACTION_TYPE:")[1].strip()
                    # Clean up any extra text after the action type
                    if action_type.startswith("CLICK"):
                        result["action"]["type"] = "CLICK"
                    elif action_type.startswith("WAIT"):
                        result["action"]["type"] = "WAIT"
                    elif action_type.startswith("BACK"):
                        result["action"]["type"] = "BACK"
                    elif action_type.startswith("EXIT"):
                        result["action"]["type"] = "EXIT"
                    else:
                        result["action"]["type"] = action_type
                    
                    found_action = True
                    logger.info(f"Found action type: {result['action']['type']}")
                    
                elif "COORDINATES:" in line:
                    if "[" in line and "]" in line:
                        coords_start = line.find("[")
                        coords_end = line.find("]", coords_start)
                        coords_part = line[coords_start:coords_end+1]
                        
                        # Parse coordinates
                        coords_str = coords_part.strip("[]")
                        if "," in coords_str:
                            coords_values = coords_str.split(",")
                        else:
                            coords_values = coords_str.split()
                        
                        # Convert to integers
                        coords = []
                        for val in coords_values:
                            clean_val = val.strip()
                            if clean_val.isdigit() or (clean_val.startswith("-") and clean_val[1:].isdigit()):
                                coords.append(int(clean_val))
                        
                        if len(coords) == 2:  # x, y
                            result["action"]["coordinates"] = coords
                            logger.info(f"Found coordinates: {coords}")
                        elif len(coords) == 4:  # Bounding box instead of point
                            # Convert bounding box to center point
                            x1, y1, x2, y2 = coords
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            result["action"]["coordinates"] = [center_x, center_y]
                            logger.info(f"Converted bounding box {coords} to center point: [{center_x}, {center_y}]")
                            
                elif "CONFIDENCE:" in line:
                    try:
                        conf_str = line.split("CONFIDENCE:")[1].strip()
                        # Handle percentages and decimal values
                        if "%" in conf_str:
                            conf_str = conf_str.replace("%", "")
                            confidence = float(conf_str) / 100.0
                        else:
                            confidence = float(conf_str)
                        
                        # Ensure confidence is in [0,1]
                        confidence = max(0.0, min(1.0, confidence))
                        result["action"]["confidence"] = confidence
                        logger.info(f"Found confidence: {confidence}")
                    except Exception as e:
                        logger.warning(f"Failed to parse confidence: {line}")
                        
                elif "REASONING:" in line:
                    reasoning = line.split("REASONING:")[1].strip()
                    # Stop parsing at the next action or section
                    if "ACTION_TYPE:" not in reasoning:
                        result["action"]["reasoning"] = reasoning
                        logger.info(f"Found reasoning: {reasoning[:50]}...")
                        
            except Exception as e:
                logger.warning(f"Failed to parse action line: {line}")
        
        # If no confidence was found but we have an action, set a default confidence
        if found_action and result["action"]["confidence"] == 0.0:
            result["action"]["confidence"] = 0.8  # Default confidence
            logger.info("No confidence found, setting default confidence: 0.8")
        
        # If we didn't get coordinates but we have UI elements, try to determine them
        if result["action"]["type"] == "CLICK" and not result["action"]["coordinates"] and result["ui_elements"]:
            # Look for confirmation-related elements first
            confirmation_keywords = ["CONFIRM", "YES", "OK", "START", "BEGIN", "BENCHMARK"]
            
            # If we're in a confirmation dialog, prioritize confirmation buttons
            if result.get("context", "").upper().find("CONFIRMATION") >= 0:
                confirmation_keywords = ["CONFIRM", "YES", "OK"] + confirmation_keywords
                logger.info("In confirmation dialog, prioritizing confirmation buttons")
            
            for element in result["ui_elements"]:
                element_name = element["name"].upper()
                if any(keyword in element_name for keyword in confirmation_keywords):
                    x1, y1, x2, y2 = element["coordinates"]
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    result["action"]["coordinates"] = [center_x, center_y]
                    if result["action"]["confidence"] == 0.0:
                        result["action"]["confidence"] = 0.9
                    logger.info(f"Auto-determined coordinates for element: {element_name} at ({center_x}, {center_y})")
                    break
            
            # If still no coordinates, use the first UI element
            if not result["action"]["coordinates"] and result["ui_elements"]:
                element = result["ui_elements"][0]
                x1, y1, x2, y2 = element["coordinates"]
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                result["action"]["coordinates"] = [center_x, center_y]
                if result["action"]["confidence"] == 0.0:
                    result["action"]["confidence"] = 0.75
                logger.info(f"Auto-determined coordinates for first element: {element['name']} at ({center_x}, {center_y})")
        
        # Final fallback: if we have UI elements but no action type, assume CLICK
        if not result["action"]["type"] and result["ui_elements"]:
            result["action"]["type"] = "CLICK"
            result["action"]["confidence"] = 0.7
            logger.info("No action type found, defaulting to CLICK")
            
            # Also set coordinates for the first element
            if not result["action"]["coordinates"]:
                element = result["ui_elements"][0]
                x1, y1, x2, y2 = element["coordinates"]
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                result["action"]["coordinates"] = [center_x, center_y]
                logger.info(f"Auto-determined coordinates for default action: {element['name']} at ({center_x}, {center_y})")
    
    def _check_benchmark_indicators(self, output_text: str, result: Dict) -> None:
        """Check for benchmark option and result indicators in the model output."""
        lower_output = output_text.lower()
        
        # Check for benchmark option
        if "benchmark" in lower_output or "fps test" in lower_output or "performance test" in lower_output:
            option_phrases = [
                "this is the target", 
                "found benchmark",
                "benchmark option",
                "benchmark button",
                "performance test button",
                "fps test option"
            ]
            
            if any(phrase in lower_output for phrase in option_phrases):
                result["is_benchmark_option"] = True
                logger.info("Benchmark option found!")
        
        # Check for benchmark results
        result_indicators = ["benchmark complete", "test finished", "results", "summary", "fps results"]
        
        if any(indicator in lower_output for indicator in result_indicators):
            result_phrases = [
                "benchmark has completed",
                "benchmark results",
                "test completed",
                "results are shown",
                "benchmark finished"
            ]
            
            if any(phrase in lower_output for phrase in result_phrases):
                result["is_benchmark_result"] = True
                logger.info("Benchmark results detected!")
    
    def create_annotated_image(self, original_path: str, analysis: Dict, output_path: str) -> None:
        """Create an annotated image with UI analysis visualization.
        
        Args:
            original_path: Path to the original screenshot
            analysis: Analysis results dictionary
            output_path: Path to save the annotated image
        """
        try:
            # Load the original image
            img = Image.open(original_path)
            draw = ImageDraw.Draw(img)
            
            # Try to load fonts
            try:
                font = ImageFont.truetype("arial.ttf", 16)
                small_font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Colors for different element types
            colors = {
                "regular": (0, 255, 0),  # Green
                "benchmark": (255, 0, 0),  # Red
                "action": (0, 0, 255),    # Blue
                "priority": (255, 165, 0)  # Orange for priority elements
            }
            
            # Priority keywords
            priority_keywords = ["OPTIONS", "SETTINGS", "GRAPHICS", "BENCHMARK", "PERFORMANCE"]
            
            # Draw UI elements
            for element in analysis["ui_elements"]:
                if "coordinates" in element:
                    x1, y1, x2, y2 = element["coordinates"]
                    
                    # Choose color based on element type
                    color = colors["regular"]
                    element_name = element["name"].upper()
                    
                    # Check if this is a benchmark-related element
                    if "BENCHMARK" in element_name:
                        color = colors["benchmark"]
                    # Check if this is a priority element
                    elif any(keyword in element_name for keyword in priority_keywords):
                        color = colors["priority"]
                    
                    # Draw rectangle
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    # Add element name
                    text_bg_bbox = [x1, max(0, y1-20), x1 + len(element["name"])*8, y1]
                    draw.rectangle(text_bg_bbox, fill=color)
                    draw.text((x1+2, max(0, y1-18)), element["name"], fill=(0, 0, 0), font=small_font)
            
            # Draw action point if exists
            if analysis["action"]["coordinates"]:
                x, y = analysis["action"]["coordinates"]
                radius = 10
                
                # Draw target circle at action point
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=None, outline=colors["action"], width=3)
                
                # Draw crosshair
                draw.line((x-15, y, x+15, y), fill=colors["action"], width=2)
                draw.line((x, y-15, x, y+15), fill=colors["action"], width=2)
                
                # Add action label
                action_label = f"CLICK: {analysis['action']['confidence']:.2f}"
                draw.text((x+15, y-10), action_label, fill=colors["action"], font=font)
            
            # Add context and action info as text at the top
            info_text = []
            if analysis["context"]:
                info_text.append(f"CONTEXT: {analysis['context']}")
            
            if analysis["action"]["type"]:
                confidence = analysis["action"]["confidence"]
                info_text.append(f"ACTION: {analysis['action']['type']} (Confidence: {confidence:.2f})")
                
            if analysis["action"]["reasoning"]:
                reasoning = analysis["action"]["reasoning"]
                # Wrap long reasoning text
                if len(reasoning) > 60:
                    wrapped_reasoning = []
                    for i in range(0, len(reasoning), 60):
                        wrapped_reasoning.append(reasoning[i:i+60])
                    for i, line in enumerate(wrapped_reasoning):
                        if i == 0:
                            info_text.append(f"REASONING: {line}")
                        else:
                            info_text.append(f"           {line}")
                else:
                    info_text.append(f"REASONING: {reasoning}")
            
            # Add heading at the top
            heading_bg = (0, 0, 0, 180)  # Semi-transparent black
            heading_height = len(info_text) * 25 + 10
            heading_rect = [0, 0, img.width, heading_height]
            
            # Create semi-transparent overlay for the header
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle(heading_rect, fill=heading_bg)
            img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(img)
            
            # Add the text at the top
            y_offset = 10
            for text in info_text:
                draw.text((10, y_offset), text, fill=(255, 255, 255), font=font)
                y_offset += 25
            
            # Save the annotated image
            img.save(output_path)
            logger.info(f"Annotated image saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating annotated image: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _cleanup_resources(self):
        """Clean up resources to avoid memory leaks."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()