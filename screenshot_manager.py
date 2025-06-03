"""
Screenshot Manager module for capturing and managing game screenshots.
"""
import os
import logging
import time
from pathlib import Path
from typing import Dict, Optional
import pyautogui
from PIL import Image

logger = logging.getLogger("ScreenshotManager")

class ScreenshotManager:
    """Manages game screenshots for the benchmarking process."""
    
    def __init__(self, directories: Dict[str, Path]):
        """Initialize the screenshot manager.
        
        Args:
            directories: Dictionary of directory paths for storing screenshots
        """
        self.directories = directories
        self.screenshot_counter = 0
    
    def take_screenshot(self) -> str:
        """Take a screenshot and save it to the raw screenshots directory.
        
        Returns:
            Path to the saved screenshot
        """
        self.screenshot_counter += 1
        screenshot_path = os.path.join(
            self.directories["raw_screenshots"], 
            f"screenshot_{self.screenshot_counter:04d}.png"
        )
        
        try:
            # Use pyautogui only for screenshot capture
            screenshot = pyautogui.screenshot()
            screenshot.save(screenshot_path)
            
            logger.info(f"Screenshot saved: {screenshot_path}")
            return screenshot_path
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return ""
    
    def save_benchmark_result(self, screenshot_path: str, result_name: Optional[str] = None) -> str:
        """Save a copy of the screenshot as a benchmark result.
        
        Args:
            screenshot_path: Path to the screenshot
            result_name: Optional custom name for the result image
            
        Returns:
            Path to the saved benchmark result image
        """
        if not result_name:
            result_name = f"benchmark_result_{int(time.time())}.png"
            
        result_path = os.path.join(self.directories["benchmark_results"], result_name)
        
        try:
            # Copy the screenshot to the benchmark results directory
            img = Image.open(screenshot_path)
            img.save(result_path)
            img.close()
            
            logger.info(f"Benchmark result saved: {result_path}")
            return result_path
        except Exception as e:
            logger.error(f"Failed to save benchmark result: {e}")
            return ""
    
    def get_screenshot_count(self) -> int:
        """Get the current screenshot count.
        
        Returns:
            Number of screenshots taken
        """
        return self.screenshot_counter
    
    def get_latest_screenshot_path(self) -> str:
        """Get the path to the latest screenshot.
        
        Returns:
            Path to the latest screenshot
        """
        if self.screenshot_counter > 0:
            return os.path.join(
                self.directories["raw_screenshots"], 
                f"screenshot_{self.screenshot_counter:04d}.png"
            )
        return ""