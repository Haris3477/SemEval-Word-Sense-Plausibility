#!/usr/bin/env python
"""Monitor training progress by checking for output files and process activity."""

import time
import subprocess
import os
from pathlib import Path

def check_process(pid):
    """Check if process is still running."""
    try:
        result = subprocess.run(['ps', '-p', str(pid)], capture_output=True)
        return result.returncode == 0
    except:
        return False

def get_runtime(pid):
    """Get process runtime."""
    try:
        result = subprocess.run(['ps', '-p', str(pid), '-o', 'etime='], 
                              capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return "unknown"

def check_output_files():
    """Check for training output files."""
    files = {}
    if Path('outputs/best_model.pt').exists():
        files['model'] = Path('outputs/best_model.pt').stat().st_mtime
    if Path('results_summary.csv').exists():
        files['results'] = Path('results_summary.csv').stat().st_mtime
    return files

def main():
    pid = 45971  # The 1-epoch training process
    print("Monitoring training progress...")
    print("=" * 60)
    
    start_time = time.time()
    last_file_time = {}
    
    while True:
        if not check_process(pid):
            print("\n✓ Training completed!")
            break
        
        runtime = get_runtime(pid)
        elapsed = time.time() - start_time
        
        # Check for new files
        files = check_output_files()
        if files:
            if 'model' in files:
                if 'model' not in last_file_time or files['model'] > last_file_time.get('model', 0):
                    print(f"\n✓ Model checkpoint saved! (at {runtime} runtime)")
                    last_file_time['model'] = files['model']
            if 'results' in files:
                if 'results' not in last_file_time or files['results'] > last_file_time.get('results', 0):
                    print(f"\n✓ Results saved! (at {runtime} runtime)")
                    last_file_time['results'] = files['results']
        
        # Show status
        print(f"\r⏳ Training running... Runtime: {runtime} | Monitoring: {int(elapsed)}s", end='', flush=True)
        
        time.sleep(5)
    
    print("\n" + "=" * 60)
    print("Training finished! Check results_summary.csv for metrics.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


