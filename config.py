"""
Configuration settings for the Game Benchmarking System.
"""
import os
from datetime import datetime
from pathlib import Path

# Base directory for benchmark runs
BASE_DIR = Path("benchmark_runs")

# Model settings
MODEL_CONFIG = {
    "model_path": "Qwen/Qwen2.5-VL-7B-Instruct",
    "device": "cuda",
    "torch_dtype": "bfloat16",
    "attn_implementation": "flash_attention_2",
    "temperature": 0.2,
    "top_p": 0.9, 
    "max_new_tokens": 2048,
    "repetition_penalty": 1.05
}

# Benchmark execution settings
BENCHMARK_CONFIG = {
    "initial_wait_time": 5,         # Seconds to wait after game launch
    "screenshot_interval": 2.0,     # Seconds between screenshots
    "max_navigation_attempts": 15,  # Maximum attempts to navigate through menus
    "benchmark_timeout": 120,       # Maximum seconds to wait for benchmark to complete
    "confidence_threshold": 0.90,   # Minimum confidence for UI actions
    "navigation_delay": 1.2         # Seconds to wait after navigation action
}

# UI detection prompt for the Qwen model
UI_PROMPT = """
You are **Game Benchmark AI**, an expert at navigating video game menus to run benchmarks and capture results automatically using visual understanding.

Your workflow is:
1) Find and initiate the benchmark
2) Wait for benchmark completion (detect results screen)
3) Capture and save the result screenshot
4) Navigate back to main menu
5) Exit to desktop gracefully

---

When analyzing the screenshot, follow this structure:

CONTEXT:
- Describe what type of screen this is (e.g., Main Menu, Options, Graphics Settings, In-Game Popup, Benchmark Results, etc.)

DETECTED UI ELEMENTS:
- [Element name]: [X1, Y1, X2, Y2] - [Short description or likely function]
  Example: "Graphics Settings": [340, 510, 600, 580] - Likely leads to video or display settings

NAVIGATION ACTION:
- ACTION_TYPE: [CLICK | BACK | WAIT | EXIT]
- COORDINATES: [X, Y] (only if ACTION_TYPE is CLICK)
- CONFIDENCE: [0.00 to 1.00]
- REASONING: [Brief explanation of why this action should be taken next]

---

### Navigation Logic:
- First, look for benchmark triggers like:  
  `"Start Benchmark"`, `"Run Benchmark"`, `"Benchmark"`, `"Begin Test"`, `"CS2 FPS BENCHMARK"`, `"GO"`, `"Performance Test"`

- Common paths to benchmarks (try in order):
  - "Options" → "Graphics" → "Benchmark"
  - "Settings" → "Video" → "Performance Test"
  - "WORKSHOP MAPS" → "CS2 FPS BENCHMARK" → "GO"
  - Don't directly click on "GO" make sure "WORKSHOP MAPS" is clicked before clicking on "GO"
  - **IMPORTANT** Don't group multiple UI buttons into a single button.
  - If unsure, explore one menu option at a time until benchmark is found

- If benchmark-related buttons are **not found**, locate a `"Back"` button and return to the previous menu.
  - Acceptable labels: `"Home"`, `"Home Icon"`, `"Back"`, `"BACK"`, `"Go Back"`, `"←"` — match case-insensitively
  - Click multiple times if needed to return to the Main Menu
  - Then explore another unvisited menu option

- If a confirmation dialog appears (e.g., "Are you sure?"), click **"Yes"**, **"OK"**, or **"Confirm"**

- Once benchmark completes (i.e., results are visible), mark benchmark as **finished**
  - Then: Save screenshot to `[timestamped_run]/results/`
  - Then: Navigate to Main Menu and EXIT the game

- If an in-game **popup or modal** appears (e.g., notification, tutorial, warning), detect it and take suitable action to dismiss (e.g., “Close”, “Skip”, “OK”, “Continue”, “Esc”)

---

### Example Hints:
- In most games, benchmark tools are located in **"Options"**, or **"Graphics Settings"**, or **"WORKSHOP MAPS"**
- If not found, explore other menus **one by one** and return if irrelevant
- Always wait for visual transitions (ensure screen has fully changed before making next decision)

---

### Restrictions:
1. Output ONLY ONE `NAVIGATION ACTION` per response
2. Never output placeholder coordinates — all CLICK actions must provide pixel (X, Y) values
3. Always log:
   - Screen context
   - Detected elements (with bboxes)
   - Navigation reasoning
   - Action executed

---

You must act like a vision-powered agent controlling the game via precise win32api commands.
"""

# Debug settings
DEBUG_CONFIG = {
    "enabled": True,
    "verbose_logging": True,
    "draw_bounding_boxes": True,
    "save_model_responses": True
}

# Keyboard shortcuts
KEYBOARD_SHORTCUTS = {
    "exit_game": "alt+f4",
    "escape_menu": "escape",
    "screenshot": "f12"
}

# Function to create a run directory structure for a new benchmark run
def create_run_directory():
    """Create a timestamped run directory structure and return paths."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = BASE_DIR / f"run_{timestamp}"
    
    # Create subdirectories
    directories = {
        "root": run_dir,
        "raw_screenshots": run_dir / "Raw Screenshots",
        "analyzed_screenshots": run_dir / "Analyzed Screenshots",
        "logs": run_dir / "Logs",
        "benchmark_results": run_dir / "Benchmark Results"
    }
    
    # Create all directories
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories