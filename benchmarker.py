"""
Main Game Benchmarker module for running automated game benchmarks.
"""
import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import subprocess

from ui_analyzer import UIAnalyzer
from input_controller import InputController
from screenshot_manager import ScreenshotManager
from flow_manager import FlowManager
from result_detector import ResultDetector

class GameBenchmarker:
    """Main class for automated game benchmarking."""
    
    def __init__(self, config_path: str = "config.py"):
        """Initialize the benchmarker with configuration.
        
        Args:
            config_path: Path to the configuration module
        """
        # Import configuration
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        
        # Initialize directories
        self.directories = config.create_run_directory()
        
        # Store configurations
        self.model_config = config.MODEL_CONFIG
        self.benchmark_config = config.BENCHMARK_CONFIG
        self.ui_prompt = config.UI_PROMPT
        self.debug_config = config.DEBUG_CONFIG
        self.keyboard_shortcuts = config.KEYBOARD_SHORTCUTS
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize components
        self.ui_analyzer = UIAnalyzer(self.model_config)
        self.input_controller = InputController(self.benchmark_config["navigation_delay"])
        self.screenshot_manager = ScreenshotManager(self.directories)
        self.flow_manager = FlowManager("flow.json")
        self.result_detector = ResultDetector(self.flow_manager)
        
        # State tracking
        self.current_attempt = 0
        self.penalties = {}
        self.game_process = None
        
        self.logger.info("GameBenchmarker initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration.
        
        Returns:
            Configured logger instance
        """
        log_file = os.path.join(self.directories["logs"], "benchmark.log")
        logging.basicConfig(
            level=logging.DEBUG if self.debug_config["verbose_logging"] else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger("GameBenchmarker")
    
    def launch_game(self, game_path: str) -> bool:
        """Launch a game executable.
        
        Args:
            game_path: Path to the game executable
            
        Returns:
            True if game launched successfully, False otherwise
        """
        if not game_path:
            self.logger.info("No game path provided, assuming game is already running")
            return True
            
        try:
            self.logger.info(f"Launching game: {game_path}")
            self.game_process = subprocess.Popen(game_path)
            
            # Wait for game to start
            self.logger.info(f"Waiting {self.benchmark_config['initial_wait_time']} seconds for game to initialize...")
            time.sleep(self.benchmark_config["initial_wait_time"])
            
            # Check if process is still running
            if self.game_process.poll() is not None:
                self.logger.error("Game process terminated unexpectedly")
                return False
                
            self.logger.info("Game launched successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to launch game: {e}")
            return False
    
    def run_benchmark(self) -> bool:
        """Run the benchmark process.
        
        Returns:
            True if benchmark completed successfully, False otherwise
        """
        self.logger.info("Starting benchmark process")
        
        try:
            # Track timing
            start_time = time.time()
            
            while True:
                # Check timeout
                elapsed_time = time.time() - start_time
                if elapsed_time > self.benchmark_config["benchmark_timeout"]:
                    self.logger.warning(f"Benchmark timed out after {elapsed_time:.1f} seconds")
                    return False
                    
                # Take screenshot
                screenshot_path = self.screenshot_manager.take_screenshot()
                if not screenshot_path:
                    self.logger.error("Failed to take screenshot")
                    continue
                
                # Analyze screenshot
                analysis = self.ui_analyzer.analyze_screenshot(screenshot_path, self.ui_prompt)
                
                # Create annotated visualization if debug enabled
                if self.debug_config["draw_bounding_boxes"]:
                    analyzed_path = os.path.join(
                        self.directories["analyzed_screenshots"], 
                        f"analyzed_{self.screenshot_manager.get_screenshot_count():04d}.png"
                    )
                    self.ui_analyzer.create_annotated_image(screenshot_path, analysis, analyzed_path)
                
                # Save model response if debug enabled
                if self.debug_config["save_model_responses"]:
                    response_path = os.path.join(
                        self.directories["logs"], 
                        f"response_{self.screenshot_manager.get_screenshot_count():04d}.txt"
                    )
                    with open(response_path, 'w') as f:
                        f.write(analysis.get("raw_text", "No response"))
                
                # Process analysis
                self._process_analysis_result(analysis, screenshot_path)
                
                # Check if benchmark is completed
                if self.result_detector.is_benchmark_completed():
                    self.logger.info("Benchmark completed successfully!")
                    
                    # Save benchmark result screenshot
                    self.screenshot_manager.save_benchmark_result(screenshot_path)
                    
                    # Navigate back to main menu and exit
                    self._navigate_back_to_main_menu()
                    return True
                
                # Check if we've reached max attempts
                self.current_attempt += 1
                if self.current_attempt >= self.benchmark_config["max_navigation_attempts"]:
                    self.logger.warning("Reached maximum navigation attempts")
                    return False
                
                # Detect navigation loops
                if self.input_controller.detect_navigation_loop():
                    self.logger.warning("Breaking out of navigation loop...")
                    # Press Escape to try to break the loop
                    self.input_controller.press_key("escape")
                
                # Wait before next screenshot
                time.sleep(self.benchmark_config["screenshot_interval"])
                
        except Exception as e:
            self.logger.error(f"Error during benchmark process: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
        finally:
            # Log benchmark summary
            self._log_benchmark_summary()
    
    def _process_analysis_result(self, analysis: Dict, screenshot_path: str) -> None:
        """Process the UI analysis result and take appropriate action.
        
        Args:
            analysis: UI analysis result
            screenshot_path: Path to the current screenshot
        """
        # Log the current context
        if analysis["context"]:
            self.logger.info(f"Current context: {analysis['context']}")
            
            # Try to detect the game based on context if not already detected
            if not self.flow_manager.current_game:
                self.flow_manager.detect_game(analysis["context"])
        
        # Log found UI elements for debugging
        if analysis["ui_elements"]:
            element_names = [elem["name"] for elem in analysis["ui_elements"]]
            self.logger.info(f"Found UI elements: {element_names}")
        
        # Get the current navigation step based on context
        navigation_step = None
        if analysis["context"]:
            navigation_step = self.flow_manager.get_navigation_step(analysis["context"])
        
        # Check for benchmark option if benchmark not started
        if not self.result_detector.is_benchmark_started():
            # First check for explicit benchmark options
            if self.result_detector.check_benchmark_option(analysis):
                self.result_detector.set_benchmark_started(True)
                self.logger.info("Benchmark option found! Clicking to start benchmark...")
            else:
                # Also check for benchmark-related UI elements directly
                benchmark_keywords = ["BENCHMARK", "TEST", "START", "BEGIN"]
                for element in analysis["ui_elements"]:
                    element_name = element["name"].upper()
                    if any(keyword in element_name for keyword in benchmark_keywords):
                        self.result_detector.set_benchmark_started(True)
                        self.logger.info(f"Benchmark element found: {element_name}")
                        break
        
        # Check for benchmark results if benchmark started
        if self.result_detector.is_benchmark_started() and not self.result_detector.is_benchmark_completed():
            if self.result_detector.check_benchmark_result(analysis):
                self.result_detector.set_benchmark_completed(True)
                self.logger.info("Benchmark results detected!")
                return  # Stop processing, we'll handle results in the main loop
        
        # Log action details for debugging
        action = analysis["action"]
        self.logger.info(f"Action: {action['type']}, Confidence: {action['confidence']}, Coordinates: {action.get('coordinates')}")
        
        # Execute the recommended action if confidence is high enough
        if action["confidence"] >= self.benchmark_config["confidence_threshold"]:
            # If we have a navigation step, override the action to follow the flow
            if navigation_step:
                target_element = self.flow_manager.get_target_element(analysis["ui_elements"], navigation_step)
                if target_element and "coordinates" in target_element:
                    x1, y1, x2, y2 = target_element["coordinates"]
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    action["coordinates"] = [center_x, center_y]
                    action["type"] = "CLICK"
                    action["confidence"] = 0.95  # High confidence for navigation path
                    self.logger.info(f"Following navigation flow: clicking on {target_element['name']}")
            
            # Special handling for confirmation dialogs
            elif analysis.get("context", "").upper().find("CONFIRMATION") >= 0 and not action.get("coordinates"):
                # Look for confirmation buttons
                confirmation_keywords = ["CONFIRM", "YES", "OK"]
                for element in analysis["ui_elements"]:
                    element_name = element["name"].upper()
                    if any(keyword in element_name for keyword in confirmation_keywords):
                        x1, y1, x2, y2 = element["coordinates"]
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        action["coordinates"] = [center_x, center_y]
                        action["type"] = "CLICK"
                        action["confidence"] = 0.95
                        self.logger.info(f"Found confirmation button: {element_name} at ({center_x}, {center_y})")
                        break
            
            # Execute the action
            success = self.input_controller.execute_action(action)
            if not success:
                self.logger.warning("Failed to execute action")
            elif action["type"] == "CLICK" and self.flow_manager.current_game:
                # Check if we should advance to the next navigation step
                self.logger.info("Action executed successfully, considering advancing navigation step")
        else:
            self.logger.warning(f"Action confidence too low: {action['confidence']}")
            
            # Try to find a suitable element to click as fallback
            target_element = self.flow_manager.get_target_element(analysis["ui_elements"])
            if target_element and "coordinates" in target_element:
                x1, y1, x2, y2 = target_element["coordinates"]
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                self.logger.info(f"Using fallback: clicking on {target_element['name']} at ({center_x}, {center_y})")
                self.input_controller.click((center_x, center_y))
    
    def _navigate_back_to_main_menu(self) -> None:
        """Navigate back to the main menu after benchmark completion."""
        self.logger.info("Navigating back to main menu...")
        
        # If we have game-specific information, use it
        if self.flow_manager.current_game:
            # Look for a "back to main menu" instruction in the game flow
            for hint in self.flow_manager.current_game.get("detection_hints", []):
                back_to_menu = hint.get("back_to_main_menu", [])
                if back_to_menu:
                    self.logger.info(f"Using game-specific back to menu instructions")
                    
                    # Execute each instruction in sequence
                    for instruction in back_to_menu:
                        action_type = instruction.get("action", "").upper()
                        
                        if action_type == "PRESS_KEY":
                            key = instruction.get("key", "")
                            self.logger.info(f"Pressing key: {key}")
                            self.input_controller.press_key(key)
                        elif action_type == "CLICK":
                            # Take a screenshot and analyze to find the element
                            screenshot_path = self.screenshot_manager.take_screenshot()
                            analysis = self.ui_analyzer.analyze_screenshot(screenshot_path, self.ui_prompt)
                            
                            # Look for elements matching the target
                            target = instruction.get("target", "").upper()
                            for element in analysis["ui_elements"]:
                                element_name = element["name"].upper()
                                if target in element_name or element_name in target:
                                    x1, y1, x2, y2 = element["coordinates"]
                                    center_x = (x1 + x2) // 2
                                    center_y = (y1 + y2) // 2
                                    self.logger.info(f"Clicking on {element['name']} to return to main menu")
                                    self.input_controller.click((center_x, center_y))
                                    break
                        
                        # Wait between actions
                        time.sleep(1)
                    
                    # After executing all instructions, wait to ensure we're at the main menu
                    time.sleep(2)
                    return
        
        # Generic fallback - press escape a few times to get back to main menu
        self.logger.info("No game-specific instructions found, using generic escape method")
        for _ in range(3):
            self.input_controller.press_key("escape")
            time.sleep(1)
    
    def exit_game(self) -> None:
        """Exit the game gracefully."""
        self.logger.info("Exiting game...")
        
        # If we have game-specific information, use it
        if self.flow_manager.current_game:
            # Look for an "exit_game" instruction in the game flow
            for hint in self.flow_manager.current_game.get("detection_hints", []):
                exit_instruction = hint.get("exit_game", {})
                if exit_instruction:
                    # Execute the exit instruction
                    action_type = exit_instruction.get("action", "").upper()
                    
                    if action_type == "PRESS_KEY":
                        key = exit_instruction.get("key", "")
                        self.logger.info(f"Pressing key: {key}")
                        self.input_controller.press_key(key)
                        return
                    elif action_type == "CLICK":
                        # Take a screenshot and analyze to find the element
                        screenshot_path = self.screenshot_manager.take_screenshot()
                        analysis = self.ui_analyzer.analyze_screenshot(screenshot_path, self.ui_prompt)
                        
                        # Look for elements matching the target
                        target = exit_instruction.get("target", "").upper()
                        for element in analysis["ui_elements"]:
                            element_name = element["name"].upper()
                            if target in element_name or element_name in target:
                                x1, y1, x2, y2 = element["coordinates"]
                                center_x = (x1 + x2) // 2
                                center_y = (y1 + y2) // 2
                                self.logger.info(f"Clicking on {element['name']} to exit game")
                                self.input_controller.click((center_x, center_y))
                                
                                # Wait for confirmation if needed
                                if exit_instruction.get("confirmation", False):
                                    time.sleep(1)
                                    confirm_screenshot = self.screenshot_manager.take_screenshot()
                                    confirm_analysis = self.ui_analyzer.analyze_screenshot(confirm_screenshot, self.ui_prompt)
                                    
                                    # Look for confirmation button
                                    confirm_target = exit_instruction.get("confirmation_target", "YES").upper()
                                    for confirm_element in confirm_analysis["ui_elements"]:
                                        confirm_name = confirm_element["name"].upper()
                                        if confirm_target in confirm_name:
                                            cx1, cy1, cx2, cy2 = confirm_element["coordinates"]
                                            c_center_x = (cx1 + cx2) // 2
                                            c_center_y = (cy1 + cy2) // 2
                                            self.logger.info(f"Clicking on confirmation: {confirm_element['name']}")
                                            self.input_controller.click((c_center_x, c_center_y))
                                            break
                                
                                return
        
        # Generic fallback - look for exit/quit elements
        screenshot_path = self.screenshot_manager.take_screenshot()
        analysis = self.ui_analyzer.analyze_screenshot(screenshot_path, self.ui_prompt)
        
        exit_keywords = ["EXIT", "QUIT", "EXIT GAME", "QUIT GAME", "EXIT TO DESKTOP"]
        for element in analysis["ui_elements"]:
            element_name = element["name"].upper()
            if any(keyword in element_name for keyword in exit_keywords):
                x1, y1, x2, y2 = element["coordinates"]
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                self.logger.info(f"Found exit element: {element['name']}")
                self.input_controller.click((center_x, center_y))
                
                # Wait and check for confirmation dialog
                time.sleep(1)
                confirm_screenshot = self.screenshot_manager.take_screenshot()
                confirm_analysis = self.ui_analyzer.analyze_screenshot(confirm_screenshot, self.ui_prompt)
                
                # Look for yes/confirm buttons
                confirm_keywords = ["YES", "CONFIRM", "OK", "EXIT"]
                for confirm_element in confirm_analysis["ui_elements"]:
                    confirm_name = confirm_element["name"].upper()
                    if any(keyword in confirm_name for keyword in confirm_keywords):
                        cx1, cy1, cx2, cy2 = confirm_element["coordinates"]
                        c_center_x = (cx1 + cx2) // 2
                        c_center_y = (cy1 + cy2) // 2
                        self.logger.info(f"Clicking on confirmation: {confirm_element['name']}")
                        self.input_controller.click((c_center_x, c_center_y))
                        break
                
                return
        
        # Last resort - use ALT+F4
        self.logger.warning("No exit elements found, using ALT+F4")
        self.input_controller.press_key("alt+f4")
        
        # Check for confirmation dialog
        time.sleep(1)
        confirm_screenshot = self.screenshot_manager.take_screenshot()
        confirm_analysis = self.ui_analyzer.analyze_screenshot(confirm_screenshot, self.ui_prompt)
        
        # Look for yes/confirm buttons
        confirm_keywords = ["YES", "CONFIRM", "OK", "EXIT"]
        for confirm_element in confirm_analysis["ui_elements"]:
            confirm_name = confirm_element["name"].upper()
            if any(keyword in confirm_name for keyword in confirm_keywords):
                cx1, cy1, cx2, cy2 = confirm_element["coordinates"]
                c_center_x = (cx1 + cx2) // 2
                c_center_y = (cy1 + cy2) // 2
                self.logger.info(f"Clicking on confirmation: {confirm_element['name']}")
                self.input_controller.click((c_center_x, c_center_y))
                break
    
    def _log_benchmark_summary(self) -> None:
        """Log benchmark execution summary."""
        self.logger.info("-" * 50)
        self.logger.info("Benchmark Summary:")
        self.logger.info(f"Total screenshots: {self.screenshot_manager.get_screenshot_count()}")
        self.logger.info(f"Benchmark started: {self.result_detector.is_benchmark_started()}")
        self.logger.info(f"Benchmark completed: {self.result_detector.is_benchmark_completed()}")
        self.logger.info(f"Navigation attempts: {self.current_attempt}")
        
        # Save summary to file
        summary_path = os.path.join(self.directories["logs"], "summary.json")
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_screenshots": self.screenshot_manager.get_screenshot_count(),
            "benchmark_started": self.result_detector.is_benchmark_started(),
            "benchmark_completed": self.result_detector.is_benchmark_completed(),
            "total_attempts": self.current_attempt,
            "navigation_history": self.input_controller.get_navigation_history()
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Summary saved to {summary_path}")
        self.logger.info("-" * 50)
    
    def __del__(self):
        """Clean up resources when the object is destroyed."""
        try:
            # Ensure game process is terminated if we started it
            if self.game_process and self.game_process.poll() is None:
                self.logger.info("Terminating game process")
                self.game_process.terminate()
                
            # Clean up resources
            if hasattr(self, 'ui_analyzer'):
                delattr(self, 'ui_analyzer')
                
            # Force garbage collection
            import gc
            gc.collect()
            
            if hasattr(self, 'logger'):
                self.logger.info("GameBenchmarker resources cleaned up")
        except:
            # Ignore errors during cleanup
            pass