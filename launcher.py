"""
Launcher module for running the Game Benchmarking System.
"""
import os
import sys
import logging
import argparse
import importlib.util
import time
import shutil
from typing import Optional

from benchmarker import GameBenchmarker

def setup_logging() -> logging.Logger:
    """Set up logging for the launcher.
    
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger("Launcher")

def verify_dependencies(logger: logging.Logger) -> bool:
    """Verify that all required dependencies are installed.
    
    Args:
        logger: Logger instance
        
    Returns:
        True if all dependencies are available, False otherwise
    """
    required_modules = [
        "torch", "transformers", "numpy", "win32api", "pyautogui", "PIL"
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            if module == "win32api":
                import win32api
            else:
                importlib.import_module(module)
            logger.info(f"✓ {module} available")
        except ImportError:
            logger.error(f"✗ {module} missing")
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"Please install missing dependencies: pip install {' '.join(missing_modules)}")
        return False
    
    return True

def verify_gpu(logger: logging.Logger) -> bool:
    """Verify that a GPU is available for CUDA operations.
    
    Args:
        logger: Logger instance
        
    Returns:
        True if GPU is available, False otherwise
    """
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            logger.info(f"✓ CUDA is available: {device_count} device(s)")
            logger.info(f"  Current device: {current_device} ({device_name})")
            
            # Test tensor creation on GPU
            test_tensor = torch.tensor([1, 2, 3], device="cuda")
            logger.info(f"  Test tensor on GPU: {test_tensor.device}")
            
            return True
        else:
            logger.warning("⚠ CUDA is not available. GPU acceleration won't work.")
            logger.warning("  The benchmarker may run very slowly without GPU acceleration.")
            return False
    except Exception as e:
        logger.error(f"✗ Error testing GPU: {e}")
        return False

def check_flow_config(logger: logging.Logger, flow_path: Optional[str] = "flow.json") -> bool:
    """Check if the flow config file exists.
    
    Args:
        logger: Logger instance
        flow_path: Path to the flow config file
        
    Returns:
        True if flow config exists, False otherwise
    """
    if not flow_path:
        flow_path = "flow.json"
        
    if os.path.exists(flow_path):
        logger.info(f"✓ Flow configuration found: {flow_path}")
        return True
    else:
        logger.warning(f"⚠ Flow configuration not found: {flow_path}")
        logger.warning("  A default flow configuration will be used.")
        
        # Create default flow.json
        try:
            from flow_manager import FlowManager
            fm = FlowManager()
            default_config = fm._create_default_flow_config()
            
            import json
            with open(flow_path, 'w') as f:
                json.dump(default_config, f, indent=2)
                
            logger.info(f"✓ Created default flow configuration: {flow_path}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to create default flow configuration: {e}")
            return False

def verify_directory_permissions(logger: logging.Logger) -> bool:
    """Verify that we have permissions to create directories and files.
    
    Args:
        logger: Logger instance
        
    Returns:
        True if permissions are available, False otherwise
    """
    try:
        # Create a test directory
        test_dir = os.path.join("test_permissions")
        os.makedirs(test_dir, exist_ok=True)
        
        # Create a test file
        test_file = os.path.join(test_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("Test permissions")
        
        # Clean up
        os.remove(test_file)
        os.rmdir(test_dir)
        
        logger.info("✓ Directory and file permissions verified")
        return True
    except Exception as e:
        logger.error(f"✗ Permission error: {e}")
        logger.error("  Please run the launcher with appropriate permissions")
        return False

def main() -> None:
    """Main entry point for the launcher."""
    logger = setup_logging()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Game Benchmarker Launcher')
    parser.add_argument('--config', type=str, default='config.py', help='Path to configuration file')
    parser.add_argument('--game', type=str, help='Game executable path (optional)')
    parser.add_argument('--flow', type=str, default='flow.json', help='Path to flow configuration file')
    parser.add_argument('--timeout', type=int, help='Override benchmark timeout (seconds)')
    parser.add_argument('--screenshot-interval', type=float, help='Override screenshot interval (seconds)')
    parser.add_argument('--verify', action='store_true', help='Run verification tests before starting')
    args = parser.parse_args()
    
    logger.info("Starting Game Benchmarker Launcher")
    
    # Run verification tests if requested
    if args.verify:
        logger.info("Running verification tests...")
        
        verified = True
        verified &= verify_dependencies(logger)
        verified &= verify_gpu(logger)
        verified &= check_flow_config(logger, args.flow)
        verified &= verify_directory_permissions(logger)
        
        if not verified:
            logger.error("Verification tests failed. Please fix the issues before proceeding.")
            sys.exit(1)
            
        logger.info("✓ All verification tests passed!")
    
    # Create the benchmark directory structure
    try:
        # Import configuration for directory setup
        spec = importlib.util.spec_from_file_location("config", args.config)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        
        # Override configuration with command line arguments
        if args.timeout:
            logger.info(f"Overriding timeout: {args.timeout} seconds")
            config.BENCHMARK_CONFIG["benchmark_timeout"] = args.timeout
            
        if args.screenshot_interval:
            logger.info(f"Overriding screenshot interval: {args.screenshot_interval} seconds")
            config.BENCHMARK_CONFIG["screenshot_interval"] = args.screenshot_interval
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Initialize the benchmarker
    try:
        benchmarker = GameBenchmarker(args.config)
        
        # Launch game if specified
        if args.game:
            if not benchmarker.launch_game(args.game):
                logger.error("Failed to launch game")
                sys.exit(1)
        else:
            logger.info("No game specified, assuming game is already running")
        
        # Run the benchmark
        logger.info("Starting benchmark process...")
        success = benchmarker.run_benchmark()
        
        # Exit the game when done
        if success:
            logger.info("Benchmark completed successfully!")
            logger.info("Exiting game...")
            benchmarker.exit_game()
        else:
            logger.warning("Benchmark did not complete successfully.")
            logger.info("Attempting to exit game...")
            benchmarker.exit_game()
    except Exception as e:
        logger.error(f"Error during benchmarking: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()