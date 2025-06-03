"""
Flow Manager module for handling game-specific navigation patterns.
"""
import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("FlowManager")

class FlowManager:
    """Manages game-specific navigation flows based on flow configuration."""
    
    def __init__(self, flow_config_path: str = "flow.json"):
        """Initialize the flow manager with game-specific flows.
        
        Args:
            flow_config_path: Path to the flow configuration JSON file
        """
        self.flow_config_path = flow_config_path
        self.flow_config = {}
        self.current_game = None
        self.current_navigation_step = 0
        self.navigation_path = []
        self._load_flow_config()
    
    def _load_flow_config(self) -> None:
        """Load the flow configuration from JSON file."""
        try:
            if os.path.exists(self.flow_config_path):
                with open(self.flow_config_path, 'r') as f:
                    self.flow_config = json.load(f)
                logger.info(f"Loaded flow configuration from {self.flow_config_path}")
                
                # Log available game flows
                if "game_flows" in self.flow_config:
                    game_names = [game.get("game_name", "Unknown") for game in self.flow_config.get("game_flows", [])]
                    logger.info(f"Available game flows: {game_names}")
            else:
                logger.warning(f"Flow configuration file not found: {self.flow_config_path}")
                self.flow_config = self._create_default_flow_config()
        except Exception as e:
            logger.error(f"Failed to load flow configuration: {e}")
            self.flow_config = self._create_default_flow_config()
    
    def _create_default_flow_config(self) -> Dict:
        """Create a default flow configuration if none is available."""
        return {
            "game_flows": [],
            "navigation_instructions": {
                "priority_keywords": ["BENCHMARK", "PERFORMANCE", "TEST", "FPS", "GRAPHICS"],
                "button_characteristics": {
                    "options_button": ["OPTIONS", "SETTINGS", "SETUP"],
                    "graphics_button": ["GRAPHICS", "VIDEO", "DISPLAY"],
                    "benchmark_button": ["BENCHMARK", "PERFORMANCE TEST", "FPS TEST"]
                }
            },
            "result_detection": {
                "benchmark_completion_indicators": [
                    "BENCHMARK COMPLETE",
                    "TEST FINISHED",
                    "RESULTS",
                    "SUMMARY",
                    "AVERAGE FPS"
                ]
            }
        }
    
    def detect_game(self, context: str) -> Optional[Dict]:
        """Detect which game is being benchmarked based on UI context.
        
        Args:
            context: Current UI context (e.g., "Main Menu")
            
        Returns:
            Detected game flow configuration or None if no match
        """
        if not context:
            return None
            
        upper_context = context.upper()
        
        for game in self.flow_config.get("game_flows", []):
            game_name = game.get("game_name", "")
            
            for hint in game.get("detection_hints", []):
                hint_context = hint.get("context", "").upper()
                
                # Check for exact match or partial match
                if upper_context == hint_context or hint_context in upper_context or upper_context in hint_context:
                    logger.info(f"Detected game: {game_name}")
                    self.current_game = game
                    self.navigation_path = hint.get("navigation_path", [])
                    self.current_navigation_step = 0
                    return game
        
        return None
    
    def get_navigation_step(self, context: str) -> Optional[Dict]:
        """Get the current navigation step based on context.
        
        Args:
            context: Current UI context
            
        Returns:
            Current navigation step or None if no matching step
        """
        if not self.current_game or not self.navigation_path:
            return None
            
        upper_context = context.upper() if context else ""
        
        # Find matching navigation step
        for i, step in enumerate(self.navigation_path):
            menu_context = step.get("menu", "").upper()
            
            if upper_context == menu_context or menu_context in upper_context or upper_context in menu_context:
                if self.current_navigation_step != i:
                    logger.info(f"Navigation step updated: {i} - {step.get('menu', '')} -> {step.get('click', '')}")
                    self.current_navigation_step = i
                return step
        
        return None
    
    def get_target_element(self, ui_elements: List[Dict], navigation_step: Optional[Dict] = None) -> Optional[Dict]:
        """Find the target UI element to click based on navigation flow.
        
        Args:
            ui_elements: List of detected UI elements
            navigation_step: Current navigation step (if None, use default priority)
            
        Returns:
            Target UI element to click or None if no suitable element found
        """
        if not ui_elements:
            return None
            
        target_element = None
        
        # STEP 1: If we have a specific navigation step, follow it first
        if navigation_step:
            target_button = navigation_step.get("click", "").upper()
            logger.info(f"Looking for target button: {target_button}")
            
            # Try to find the exact button to click
            for element in ui_elements:
                element_name = element["name"].upper()
                
                # Try exact match first
                if element_name == target_button:
                    target_element = element
                    logger.info(f"Found exact match for {target_button}")
                    break
                    
                # Try partial match
                if target_button in element_name or any(word in element_name for word in target_button.split()):
                    target_element = element
                    logger.info(f"Found partial match for {target_button}: {element_name}")
                    break
        
        # STEP 2: If no target found, use priority list
        if not target_element:
            # Get priority elements
            priority_elements = self._get_priority_elements()
            
            logger.info(f"Using priority list to find target: {priority_elements}")
            for priority in priority_elements:
                for element in ui_elements:
                    element_name = element["name"].upper()
                    if priority in element_name:
                        target_element = element
                        logger.info(f"Found priority element: {element_name} (priority: {priority})")
                        break
                if target_element:
                    break
        
        # STEP 3: If still no target, find any clickable element as fallback
        if not target_element:
            logger.info("Using fallback strategy to find clickable element")
            fallback_keywords = ["PLAY", "OPTIONS", "SETTINGS", "GRAPHICS", "MENU", "BUTTON"]
            
            # Add any available button characteristics from flow config
            if "navigation_instructions" in self.flow_config:
                button_chars = self.flow_config["navigation_instructions"].get("button_characteristics", {})
                for char_list in button_chars.values():
                    fallback_keywords.extend(char_list)
            
            for keyword in fallback_keywords:
                for element in ui_elements:
                    element_name = element["name"].upper()
                    if keyword in element_name:
                        target_element = element
                        logger.info(f"Found fallback element: {element_name} (keyword: {keyword})")
                        break
                if target_element:
                    break
        
        # STEP 4: Ultimate fallback: just pick the first element
        if not target_element and ui_elements:
            target_element = ui_elements[0]
            logger.info(f"No target found, using first element: {target_element.get('name', 'Unknown')}")
        
        return target_element
    
    def _get_priority_elements(self) -> List[str]:
        """Get priority elements list from flow configuration.
        
        Returns:
            List of priority element keywords
        """
        # Try to get game-specific priority elements
        if self.current_game:
            for hint in self.current_game.get("detection_hints", []):
                priority_elements = hint.get("priority_elements", [])
                if priority_elements:
                    return priority_elements
        
        # Fallback to global priority elements
        if "navigation_instructions" in self.flow_config:
            priority_elements = self.flow_config["navigation_instructions"].get("priority_keywords", [])
            if priority_elements:
                return priority_elements
        
        # Default priority elements
        return ["BENCHMARK", "PERFORMANCE", "OPTIONS", "SETTINGS", "GRAPHICS"]
    
    def get_benchmark_indicators(self) -> List[str]:
        """Get benchmark indicator keywords from flow configuration.
        
        Returns:
            List of benchmark indicator keywords
        """
        indicators = ["BENCHMARK", "FPS TEST", "PERFORMANCE TEST"]
        
        # Add game-specific indicators
        if self.current_game:
            for hint in self.current_game.get("detection_hints", []):
                if "benchmark_indicators" in hint:
                    indicators.extend(hint.get("benchmark_indicators", []))
        
        return list(set([ind.upper() for ind in indicators]))
    
    def get_result_indicators(self) -> List[str]:
        """Get benchmark result indicator keywords from flow configuration.
        
        Returns:
            List of benchmark result indicator keywords
        """
        indicators = ["BENCHMARK COMPLETE", "TEST FINISHED", "RESULTS", "SUMMARY"]
        
        # Add global indicators
        if "result_detection" in self.flow_config:
            indicators.extend(self.flow_config["result_detection"].get("benchmark_completion_indicators", []))
        
        return list(set([ind.upper() for ind in indicators]))
    
    def advance_step(self) -> None:
        """Advance to the next navigation step."""
        if self.current_navigation_step < len(self.navigation_path) - 1:
            self.current_navigation_step += 1
            step = self.navigation_path[self.current_navigation_step]
            logger.info(f"Advanced to step {self.current_navigation_step}: {step.get('menu', '')} -> {step.get('click', '')}")
    
    def reset(self) -> None:
        """Reset the flow manager state."""
        self.current_game = None
        self.current_navigation_step = 0
        self.navigation_path = []