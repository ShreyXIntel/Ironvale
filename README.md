# Automated Game Benchmarking System

An AI-powered automated benchmarking system that uses computer vision and natural language processing to navigate game menus, execute benchmarks, and capture results without human intervention.

## 🎯 Overview

This system leverages the **Qwen 2.5 Vision-Language Model** to understand game user interfaces and automatically:
- Navigate through complex game menus
- Locate and initiate benchmark tests
- Wait for benchmark completion
- Capture and save benchmark results
- Return to main menu and exit gracefully

Perfect for **performance testing, hardware validation, and automated QA workflows** in gaming and graphics industries.

## 🚀 Key Features

### 🤖 **AI-Powered Navigation**
- Uses Qwen 2.5-VL-7B model for real-time UI understanding
- Intelligent menu navigation with confidence scoring
- Game-specific flow detection and adaptation
- Automatic loop detection and recovery

### 🎮 **Multi-Game Support**
Pre-configured support for popular games:
- **Counter-Strike 2** - Workshop Maps → CS2 FPS Benchmark
- **Cyberpunk 2077** - Settings → Graphics → Benchmark
- **Far Cry 6** - Options → Graphics → Benchmark  
- **Assassin's Creed Series** - Options → Graphics → Benchmark
- **Black Myth: Wukong** - Direct benchmark access
- **Extensible framework** for adding new games

### 📊 **Comprehensive Result Capture**
- Automated screenshot capture throughout process
- Benchmark result detection and archival
- Detailed execution logs with timestamps
- Navigation history tracking
- Debug visualizations with bounding boxes

### 🛠️ **Advanced Configuration**
- GPU-accelerated inference (CUDA/Flash Attention 2)
- Configurable timeouts and confidence thresholds
- Debug modes with visual annotations
- Flexible input handling (mouse/keyboard automation)

## 📋 Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended: RTX 3060 or better)
- **RAM**: 16GB+ (model requires ~8GB VRAM)
- **Storage**: 20GB+ free space for model and results
- **OS**: Windows 10/11 (uses win32api for input control)

### Software Requirements
- **Python**: 3.8 or higher
- **CUDA**: 11.8 or higher
- **Git LFS**: For downloading large model files

## 🔧 Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-username/automated-game-benchmarker.git
cd automated-game-benchmarker
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Model (Automatic)
The Qwen 2.5-VL-7B model will be automatically downloaded from Hugging Face on first run (~15GB).

### 5. Verify Installation
```bash
python launcher.py --verify
```

## 🚀 Quick Start

### Basic Usage
```bash
# Launch with game already running
python launcher.py

# Launch specific game and benchmark
python launcher.py --game "C:\Games\Cyberpunk2077\bin\x64\Cyberpunk2077.exe"

# Custom configuration
python launcher.py --config custom_config.py --timeout 300
```

### Command Line Options
```bash
python launcher.py [OPTIONS]

Options:
  --config PATH          Configuration file (default: config.py)
  --game PATH           Game executable path (optional)
  --flow PATH           Flow configuration file (default: flow.json)
  --timeout SECONDS     Override benchmark timeout
  --screenshot-interval SECONDS  Override screenshot interval
  --verify              Run verification tests before starting
```

## 📁 Project Structure

```
automated-game-benchmarker/
├── 📜 benchmarker.py          # Main orchestration logic
├── ⚙️ config.py               # Configuration settings
├── 🎮 flow.json               # Game-specific navigation flows
├── 🗺️ flow_manager.py         # Navigation flow management
├── 🖱️ input_controller.py     # Mouse/keyboard automation
├── 🚀 launcher.py             # Entry point and verification
├── 🔍 result_detector.py      # Benchmark completion detection
├── 📸 screenshot_manager.py   # Screenshot capture and management
├── 👁️ ui_analyzer.py          # AI-powered UI analysis
├── 📝 requirements.txt        # Python dependencies
└── 📚 README.md              # This file

# Generated during execution:
benchmark_runs/
├── run_20241203_143022/       # Timestamped run directory
│   ├── Raw Screenshots/       # Original game screenshots
│   ├── Analyzed Screenshots/  # AI-annotated images
│   ├── Benchmark Results/     # Final benchmark screenshots
│   └── Logs/                 # Execution logs and summaries
```

## ⚙️ Configuration

### Model Configuration (`config.py`)
```python
MODEL_CONFIG = {
    "model_path": "Qwen/Qwen2.5-VL-7B-Instruct",
    "device": "cuda",                    # Use "cpu" if no GPU
    "torch_dtype": "bfloat16",          # Memory optimization
    "temperature": 0.2,                  # Lower = more deterministic
    "max_new_tokens": 2048,             # Response length limit
}
```

### Benchmark Settings
```python
BENCHMARK_CONFIG = {
    "initial_wait_time": 5,             # Seconds after game launch
    "screenshot_interval": 2.0,         # Time between screenshots
    "max_navigation_attempts": 15,      # Maximum menu navigation tries
    "benchmark_timeout": 120,           # Max benchmark duration
    "confidence_threshold": 0.90,       # Minimum AI confidence for actions
}
```

### Debug Options
```python
DEBUG_CONFIG = {
    "verbose_logging": True,            # Detailed log output
    "draw_bounding_boxes": True,        # Visual UI element detection
    "save_model_responses": True,       # Save AI responses to files
}
```

## 🎮 Adding New Games

### 1. Update `flow.json`
Add game-specific navigation patterns:

```json
{
  "game_name": "Your Game Name",
  "detection_hints": [{
    "context": "Main Menu",
    "priority_elements": ["OPTIONS", "SETTINGS", "BENCHMARK"],
    "navigation_path": [
      {"menu": "Main Menu", "click": "OPTIONS"},
      {"menu": "Options Menu", "click": "GRAPHICS"},
      {"menu": "Graphics Settings", "click": "BENCHMARK"}
    ],
    "benchmark_indicators": ["FPS TEST", "PERFORMANCE"],
    "back_to_main_menu": [
      {"action": "PRESS_KEY", "key": "escape"}
    ]
  }]
}
```

### 2. Test Navigation
```bash
python launcher.py --game "path/to/your/game.exe"
```

### 3. Refine Detection
Monitor logs and adjust:
- UI element names in `priority_elements`
- Navigation paths in `navigation_path`
- Result indicators in `benchmark_indicators`

## 🔍 Troubleshooting

### Common Issues

**❌ CUDA Out of Memory**
```python
# In config.py, reduce model precision:
MODEL_CONFIG["torch_dtype"] = "float16"  # or "int8"
```

**❌ Game Not Responding to Clicks**
- Ensure game is in **windowed or borderless mode**
- Check if game requires **administrator privileges**
- Verify **correct screen resolution** in screenshots

**❌ UI Elements Not Detected**
- Enable debug mode: `DEBUG_CONFIG["draw_bounding_boxes"] = True`
- Check annotated screenshots in `Analyzed Screenshots/`
- Adjust `confidence_threshold` in config

**❌ Navigation Loops**
- System automatically detects and breaks loops
- Check `flow.json` for correct navigation paths
- Monitor logs for repeated actions

### Performance Optimization

**🚀 Speed Up Inference**
```python
# Enable Flash Attention 2 (if supported)
MODEL_CONFIG["attn_implementation"] = "flash_attention_2"

# Reduce screenshot interval
BENCHMARK_CONFIG["screenshot_interval"] = 1.0
```

**💾 Reduce Memory Usage**
```python
# Use smaller model variant or quantization
MODEL_CONFIG["torch_dtype"] = "int8"  # Significant memory reduction
```

## 📊 Output Analysis

### Screenshot Organization
- **Raw Screenshots**: Original game captures with timestamps
- **Analyzed Screenshots**: AI-annotated images showing detected UI elements
- **Benchmark Results**: Final result screens automatically saved

### Log Analysis
```bash
# View execution summary
cat benchmark_runs/run_*/Logs/summary.json

# Check detailed logs
tail -f benchmark_runs/run_*/Logs/benchmark.log
```

### Navigation History
Each run includes complete navigation history with:
- Action types (CLICK, KEY_PRESS)
- Coordinates and timestamps
- UI detection confidence scores
- Context transitions

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black *.py
```

### Adding Features
1. **New Games**: Update `flow.json` with navigation patterns
2. **UI Improvements**: Enhance detection in `ui_analyzer.py`
3. **Input Methods**: Extend `input_controller.py` for new interaction types
4. **Result Processing**: Add analysis capabilities to `result_detector.py`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Qwen Team** for the excellent vision-language model
- **Hugging Face** for model hosting and transformers library
- **Game Developers** for creating benchmarkable titles
- **Open Source Community** for supporting automated testing tools

## 📞 Support

- **Issues**: Report bugs via [GitHub Issues](https://github.com/your-username/automated-game-benchmarker/issues)
- **Discussions**: Join conversations in [GitHub Discussions](https://github.com/your-username/automated-game-benchmarker/discussions)
- **Documentation**: Check the [Wiki](https://github.com/your-username/automated-game-benchmarker/wiki) for detailed guides

---

**⚡ Ready to automate your game benchmarking workflow? Get started with the Quick Start guide above!**