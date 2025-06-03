"""
Input Controller module for handling mouse and keyboard interactions.
"""
import logging
import time
from typing import Tuple, Optional, List, Dict
import win32api
import win32con
import win32gui

logger = logging.getLogger("InputController")

class InputController:
    """Handles mouse and keyboard interactions with games."""
    
    def __init__(self, navigation_delay: float = 1.0):
        """Initialize the input controller.
        
        Args:
            navigation_delay: Delay after navigation actions in seconds
        """
        self.navigation_delay = navigation_delay
        self.navigation_history = []
    
    def click(self, coordinates: Tuple[int, int], smooth: bool = True) -> bool:
        """Perform a mouse click at the specified coordinates.
        
        Args:
            coordinates: (x, y) coordinates to click
            smooth: Whether to move the cursor smoothly
            
        Returns:
            True if the click was successful, False otherwise
        """
        try:
            x, y = coordinates
            logger.info(f"Clicking at coordinates: ({x}, {y})")
            
            # Get foreground window handle to ensure we're clicking in the right window
            hwnd = win32gui.GetForegroundWindow()
            
            if smooth:
                self._move_cursor_smoothly(x, y)
            else:
                win32api.SetCursorPos((x, y))
                time.sleep(0.1)  # Short pause before clicking
            
            # Simulate mouse click
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            time.sleep(0.1)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            
            # Record the action
            self.navigation_history.append({
                "action": "CLICK",
                "coordinates": (x, y),
                "timestamp": time.time()
            })
            
            # Wait after navigation action
            time.sleep(self.navigation_delay)
            return True
        except Exception as e:
            logger.error(f"Failed to perform click: {e}")
            return False
    
    def press_key(self, key: str) -> bool:
        """Press a key or key combination.
        
        Args:
            key: Key to press (e.g., 'escape', 'alt+f4')
            
        Returns:
            True if the key press was successful, False otherwise
        """
        try:
            logger.info(f"Pressing key: {key}")
            
            if key.lower() == "escape":
                win32api.keybd_event(win32con.VK_ESCAPE, 0, 0, 0)
                time.sleep(0.1)
                win32api.keybd_event(win32con.VK_ESCAPE, 0, win32con.KEYEVENTF_KEYUP, 0)
                
                self.navigation_history.append({
                    "action": "KEY_PRESS",
                    "key": "escape",
                    "timestamp": time.time()
                })
                
            elif key.lower() == "alt+f4":
                win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)  # Alt down
                win32api.keybd_event(win32con.VK_F4, 0, 0, 0)    # F4 down
                time.sleep(0.1)
                win32api.keybd_event(win32con.VK_F4, 0, win32con.KEYEVENTF_KEYUP, 0)    # F4 up
                win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)  # Alt up
                
                self.navigation_history.append({
                    "action": "KEY_PRESS",
                    "key": "alt+f4",
                    "timestamp": time.time()
                })
                
            elif key.lower() == "enter":
                win32api.keybd_event(win32con.VK_RETURN, 0, 0, 0)
                time.sleep(0.1)
                win32api.keybd_event(win32con.VK_RETURN, 0, win32con.KEYEVENTF_KEYUP, 0)
                
                self.navigation_history.append({
                    "action": "KEY_PRESS",
                    "key": "enter",
                    "timestamp": time.time()
                })
            
            # Wait after navigation action
            time.sleep(self.navigation_delay)
            return True
        except Exception as e:
            logger.error(f"Failed to press key: {e}")
            return False
    
    def execute_action(self, action: Dict) -> bool:
        """Execute an action based on the analysis result.
        
        Args:
            action: Action dictionary from UI analysis
            
        Returns:
            True if the action was successful, False otherwise
        """
        if not action.get("type"):
            logger.warning("No action type specified")
            return False
        
        action_type = action["type"].upper()
        
        if action_type == "CLICK" and action.get("coordinates"):
            return self.click(action["coordinates"])
        elif action_type == "CLICK" and not action.get("coordinates"):
            logger.warning("CLICK action specified but no coordinates provided")
            return False
        elif action_type == "BACK":
            logger.info("Executing BACK action (pressing ESC key)")
            return self.press_key("escape")
        elif action_type == "WAIT":
            wait_time = 5  # Default wait time in seconds
            logger.info(f"Executing WAIT action for {wait_time} seconds")
            time.sleep(wait_time)
            return True
        elif action_type == "EXIT":
            logger.info("Executing EXIT action (pressing ALT+F4)")
            return self.press_key("alt+f4")
        else:
            logger.warning(f"Unknown action type: {action_type}")
            return False
    
    def _move_cursor_smoothly(self, target_x: int, target_y: int, steps: int = 10) -> None:
        """Move the cursor smoothly to target coordinates.
        
        Args:
            target_x: Target X coordinate
            target_y: Target Y coordinate
            steps: Number of steps for smooth movement
        """
        current_x, current_y = win32api.GetCursorPos()
        
        # Calculate move steps
        x_step = (target_x - current_x) / steps
        y_step = (target_y - current_y) / steps
        
        # Move cursor gradually
        for i in range(steps):
            new_x = int(current_x + x_step * (i + 1))
            new_y = int(current_y + y_step * (i + 1))
            win32api.SetCursorPos((new_x, new_y))
            time.sleep(0.02)  # Short delay between movements
        
        # Ensure final position is exactly what we want
        win32api.SetCursorPos((target_x, target_y))
        time.sleep(0.1)  # Small pause before next action
    
    def detect_navigation_loop(self, max_history: int = 6) -> bool:
        """Detect if we're stuck in a navigation loop.
        
        Args:
            max_history: Number of recent actions to check for loops
            
        Returns:
            True if a navigation loop is detected, False otherwise
        """
        if len(self.navigation_history) < max_history:
            return False
            
        # Check last N actions for a repeating pattern of N/2
        last_n = self.navigation_history[-max_history:]
        pattern_size = max_history // 2
        
        pattern_1 = [a["action"] for a in last_n[:pattern_size]]
        pattern_2 = [a["action"] for a in last_n[pattern_size:2*pattern_size]]
        
        if pattern_1 == pattern_2:
            logger.warning(f"Navigation loop detected! Pattern: {pattern_1}")
            return True
            
        return False
    
    def get_navigation_history(self) -> List[Dict]:
        """Get the navigation action history.
        
        Returns:
            List of navigation action dictionaries
        """
        return self.navigation_history