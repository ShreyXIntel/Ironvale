"""
Result Detector module for detecting benchmark completion.
"""
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("ResultDetector")

class ResultDetector:
    """Detects benchmark completion and result screens."""
    
    def __init__(self, flow_manager):
        """Initialize the result detector.
        
        Args:
            flow_manager: Flow manager instance for result indicators
        """
        self.flow_manager = flow_manager
        self.benchmark_started = False
        self.benchmark_completed = False
    
    def check_benchmark_option(self, analysis: Dict) -> bool:
        """Check if the current screen contains a benchmark option.
        
        Args:
            analysis: UI analysis result
            
        Returns:
            True if benchmark option detected, False otherwise
        """
        # First check the explicit flag
        if analysis.get("is_benchmark_option", False):
            logger.info("Benchmark option explicitly detected")
            return True
        
        # Check context for confirmation dialog
        if analysis.get("context"):
            context = analysis.get("context", "").upper()
            if "CONFIRMATION" in context or "SURE" in context or "RUN BENCHMARK" in context:
                logger.info(f"Benchmark confirmation dialog detected: {context}")
                return True
        
        # Check if any UI element name contains benchmark indicators
        benchmark_indicators = self.flow_manager.get_benchmark_indicators()
        
        for element in analysis.get("ui_elements", []):
            element_name = element.get("name", "").upper()
            
            for indicator in benchmark_indicators:
                if indicator in element_name:
                    logger.info(f"Benchmark option detected in element: {element_name}")
                    return True
        
        # Check context
        if analysis.get("context"):
            context = analysis.get("context", "").upper()
            for indicator in benchmark_indicators:
                if indicator in context:
                    logger.info(f"Benchmark option detected in context: {context}")
                    return True
        
        # Check for confirmation buttons when benchmark is already started
        if self.benchmark_started:
            confirmation_keywords = ["CONFIRM", "YES", "OK", "START"]
            for element in analysis.get("ui_elements", []):
                element_name = element.get("name", "").upper()
                for keyword in confirmation_keywords:
                    if keyword in element_name:
                        logger.info(f"Benchmark confirmation button detected: {element_name}")
                        return True
        
        return False
    
    def check_benchmark_result(self, analysis: Dict) -> bool:
        """Check if the current screen shows benchmark results.
        
        Args:
            analysis: UI analysis result
            
        Returns:
            True if benchmark results detected, False otherwise
        """
        # First check the explicit flag
        if analysis.get("is_benchmark_result", False):
            logger.info("Benchmark results explicitly detected")
            return True
        
        # Check if context contains result indicators
        if analysis.get("context"):
            context = analysis.get("context", "").upper()
            result_indicators = self.flow_manager.get_result_indicators()
            
            for indicator in result_indicators:
                if indicator in context:
                    logger.info(f"Benchmark results detected in context: {context}")
                    return True
        
        # Check the raw text for result indicators
        if analysis.get("raw_text"):
            raw_text = analysis.get("raw_text", "").upper()
            result_phrases = [
                "BENCHMARK HAS COMPLETED",
                "BENCHMARK RESULTS",
                "TEST COMPLETED",
                "RESULTS ARE SHOWN",
                "BENCHMARK FINISHED",
                "AVERAGE FPS",
                "PERFORMANCE SCORE"
            ]
            
            for phrase in result_phrases:
                if phrase in raw_text:
                    logger.info(f"Benchmark results detected in text: {phrase}")
                    return True
        
        return False
    
    def set_benchmark_started(self, started: bool = True) -> None:
        """Set the benchmark started state.
        
        Args:
            started: Whether the benchmark has started
        """
        if started and not self.benchmark_started:
            logger.info("Benchmark started")
        
        self.benchmark_started = started
    
    def set_benchmark_completed(self, completed: bool = True) -> None:
        """Set the benchmark completed state.
        
        Args:
            completed: Whether the benchmark has completed
        """
        if completed and not self.benchmark_completed:
            logger.info("Benchmark completed")
        
        self.benchmark_completed = completed
    
    def is_benchmark_started(self) -> bool:
        """Check if the benchmark has started.
        
        Returns:
            True if benchmark has started, False otherwise
        """
        return self.benchmark_started
    
    def is_benchmark_completed(self) -> bool:
        """Check if the benchmark has completed.
        
        Returns:
            True if benchmark has completed, False otherwise
        """
        return self.benchmark_completed
    
    def reset(self) -> None:
        """Reset the result detector state."""
        self.benchmark_started = False
        self.benchmark_completed = False