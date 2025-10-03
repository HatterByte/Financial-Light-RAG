import gc
import psutil
from typing import Dict
from logger import logger



def get_memory_usage():
    """Get current memory usage in MB"""
    return psutil.Process().memory_info().rss / 1024 / 1024

def cleanup_memory():
    """Force garbage collection"""
    gc.collect()

class MemoryMonitor:
    """Advanced memory monitor with adaptive cleanup"""
    def __init__(self, max_memory_mb: int = 2048):
        self.max_memory_mb = max_memory_mb
        self.start_memory = get_memory_usage()
        self.cleanup_count = 0
        
    def check_and_cleanup(self) -> Dict[str, float]:
        """Check memory and cleanup if needed"""
        current = get_memory_usage()
        
        if current > self.max_memory_mb * 0.8:  # Start cleanup at 80%
            logger.warning(f"Memory high: {current:.1f}MB, cleaning up...")
            cleanup_memory()
            self.cleanup_count += 1
            current = get_memory_usage()
            
            if current > self.max_memory_mb:
                logger.error(f"Memory still high after cleanup: {current:.1f}MB")
        
        return {
            "current_mb": current,
            "usage_percent": (current / self.max_memory_mb) * 100,
            "cleanup_count": self.cleanup_count
        }