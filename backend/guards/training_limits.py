import pandas as pd
import os

class TrainingLimits:
    MAX_CSV_ROWS = 20000
    MAX_EPISODES = 5000
    MIN_EPISODES = 10
    MAX_FILE_SIZE_MB = 10
    
    @staticmethod
    def check_training_request(filepath, params):
        """
        Validate training request against resource limits.
        
        Args:
            filepath (str): Path to CSV data
            params (dict): Training parameters
            
        Returns:
            tuple: (passed: bool, reason: str)
        """
        # 1. Check File Size
        try:
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            if size_mb > TrainingLimits.MAX_FILE_SIZE_MB:
                return False, f"File size {size_mb:.2f}MB exceeds limit of {TrainingLimits.MAX_FILE_SIZE_MB}MB"
        except FileNotFoundError:
            return False, "Data file not found"
            
        # 2. Check Row Count
        try:
            # Read only header and count rows basically (or read full if small enough)
            # Efficient row count
            with open(filepath) as f:
                row_count = sum(1 for _ in f) - 1 # Minus header
                
            if row_count > TrainingLimits.MAX_CSV_ROWS:
                return False, f"Row count {row_count} exceeds limit of {TrainingLimits.MAX_CSV_ROWS}"
            if row_count < 50:
                 return False, f"Row count {row_count} is too small for training (min 50)"
        except Exception as e:
            return False, f"Failed to read file: {str(e)}"
            
        # 3. Check Parameters
        episodes = int(params.get('episodes', 0))
        if episodes > TrainingLimits.MAX_EPISODES:
             return False, f"Episodes {episodes} exceeds limit of {TrainingLimits.MAX_EPISODES}"
        if episodes < TrainingLimits.MIN_EPISODES:
             return False, f"Episodes {episodes} is too low (min {TrainingLimits.MIN_EPISODES})"
             
        return True, "Checks passed"
