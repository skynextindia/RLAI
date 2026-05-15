
import traceback
import sys
import os

try:
    # Add current dir to path
    sys.path.append(os.getcwd())
    
    # Lazy imports to find the crash
    print("Importing main components...", flush=True)
    from main import load_config, run_train
    
    print("Loading config...", flush=True)
    config = load_config()
    
    print("Starting Phase 4...", flush=True)
    run_train("BTCUSDm", config)
    
except Exception as e:
    print("\nCRASH DETECTED:")
    traceback.print_exc()
    sys.exit(1)
