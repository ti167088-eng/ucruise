#!/usr/bin/env python3
"""
Main entry point for the route assignment system
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from orchestrator import main

if __name__ == "__main__":
    # Clean up old logs
    logs_dir = project_root / "logs"
    if logs_dir.exists():
        import glob
        import time

        # Remove log files older than 7 days
        log_files = glob.glob(str(logs_dir / "*.log"))
        current_time = time.time()
        removed_count = 0

        for log_file in log_files:
            file_age = current_time - os.path.getmtime(log_file)
            if file_age > 7 * 24 * 3600:  # 7 days in seconds
                try:
                    os.remove(log_file)
                    removed_count += 1
                except OSError:
                    pass

        if removed_count > 0:
            print(f"🧹 Cleared {removed_count} old log files from logs/")

    main()