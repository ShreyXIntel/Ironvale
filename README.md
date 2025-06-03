# Automated Game Benchmarking System

An intelligent automated system for running game benchmarks using AI (Qwen 2.5 VL) for UI navigation and analysis.

## Overview

This system automatically navigates through game menus to locate and run built-in benchmarks, captures the results, and returns to the main menu before gracefully exiting. It uses the Qwen 2.5 VL vision-language model to analyze game UI elements and make navigation decisions.

## Features

- **Fully Automated:** Handles the entire benchmarking process without human intervention
- **Game-Specific Navigation:** Uses flow configuration for specialized handling of different games
- **AI-Powered UI Analysis:** Leverages Qwen 2.5 VL to detect and understand UI elements
- **Modular Design:** Separate components for UI analysis, input handling, screenshot management, etc.
- **Detailed Logging:** Comprehensive logging and visualization of the entire process
- **Graceful Game Exits:** Properly exits games after benchmarking

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- Transformers library
- Win32 API (for Windows)
- PIL (Pillow)
- pyautogui
- Games with built-in benchmarking functionality

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/game-benchmarker.git
   cd game-benchmarker
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. (Optional) Verify the installation:
   ```
   python launcher.py --verify
   ```

## Usage

### Basic Usage

```
python launcher.py
```

This assumes a game is already running and will attempt to find and run its benchmark.

### Launch a Specific Game

```
python launcher.py --game "C:\Program Files (x86)\Steam\steamapps\common\Black Myth Wukong Benchmark Tool\b1_benchmark.exe"
```

This will launch the specified game and then start the benchmarking process.

### Additional Options

```
python launcher.py --game "C:\Path\to\game.exe" --timeout 600 --screenshot-interval 1.5
```

- `--config`: Path to configuration file (default: config.py)
- `--game`: Path to game executable (optional)
- `--flow`: Path to flow configuration file (default: flow.json)
- `--timeout`: Override benchmark timeout in seconds
- `--screenshot-interval`: Override screenshot interval in seconds
- `--verify`: Run verification tests before starting

## Directory Structure

After running the benchmarker, a timestamped directory is created under `benchmark_runs/` with the following structure:

```
benchmark_runs/
└── run_YYYYMMDD_HHMMSS/
    ├── Raw Screenshots/
    ├── Analyzed Screenshots/
    ├── Logs/
    └── Benchmark Results/
```

- **Raw Screenshots**: All screenshots taken during the run
- **Analyzed Screenshots**: Visualizations of UI analysis
- **Logs**: Detailed logs and summary information
- **Benchmark Results**: Screenshots of benchmark result screens

## Configuration

### config.py

This file contains all the configuration settings for the benchmarker:

- Model settings (path, temperature, etc.)
- Benchmark execution settings (timeouts, intervals, etc.)
- UI detection prompt for the model
- Debug settings
- Keyboard shortcuts

### flow.json

This file contains game-specific navigation flows:

- Detection hints for identifying games
- Navigation paths through menus
- Priority elements for each game
- Benchmark indicators to look for
- Instructions for returning to main menu
- Instructions for exiting the game