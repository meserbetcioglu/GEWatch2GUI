#!/usr/bin/env python3
"""
GEWatch2 GUI Launcher - Starts GUI + Price Updater (closes together)
"""
import subprocess
import os
import sys
import time
import signal
import datetime

updater_process = None

def cleanup(signum=None, frame=None):
    """Kill the updater when GUI closes"""
    global updater_process
    if updater_process:
        try:
            print("\n[INFO] Closing price updater...")
            updater_process.terminate()
            updater_process.wait(timeout=5)
        except Exception as e:
            try:
                updater_process.kill()
            except:
                pass

# Register cleanup on exit
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

def main():
    global updater_process
    script_dir = os.path.dirname(os.path.abspath(__file__)) if not getattr(sys, 'frozen', False) else os.path.dirname(sys.executable)
    gui_exe = os.path.join(script_dir, 'ge_gui.exe')
    updater_exe = os.path.join(script_dir, 'ge_price_update.exe')
    
    print(f"Script directory: {script_dir}")
    
    # Start price updater in background (continuous updates)
    print(f"\n[INFO] Starting price updater...")
    try:
        with open(os.path.join(script_dir, 'updater.log'), 'w') as log_file:
            log_file.write(f"=== Price Updater Started {datetime.datetime.now().isoformat()} ===\n")
            log_file.flush()
            
            updater_process = subprocess.Popen(
                [sys.executable, 'GE Price Update.py'],
                cwd=script_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT
            )
        print(f"✓ Price updater started (PID: {updater_process.pid})")
        time.sleep(2)
    except Exception as e:
        print(f"Warning: Could not start price updater: {e}")
        updater_process = None
    
    # Run GUI (blocking)
    print(f"\n[INFO] Starting GUI application...")
    try:
        if getattr(sys, 'frozen', False) and os.path.exists(gui_exe):
            print(f"Running from: {gui_exe}")
            gui_process = subprocess.run([gui_exe], cwd=script_dir)
        else:
            print("Running Python script")
            gui_process = subprocess.run([sys.executable, 'analyze_advanced_gui.py'], cwd=script_dir)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up updater when GUI closes
        cleanup()
        print("[INFO] Goodbye!")

if __name__ == '__main__':
    main()
