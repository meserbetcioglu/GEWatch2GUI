#!/usr/bin/env python3
"""
GEWatch2 GUI Launcher - Starts GUI + Price Updater (closes together)

Usage
-----
  python launcher.py                 — Start main GUI only (default)
  python launcher.py --portfolio     — Start portfolio manager only (port 8051)
  python launcher.py --both          — Start main GUI + portfolio manager
"""
import subprocess
import os
import sys
import time
import signal
import datetime

updater_process = None
portfolio_process = None
portfolio_api_process = None


def cleanup(signum=None, frame=None):
    """Kill background processes when GUI closes."""
    global updater_process, portfolio_process, portfolio_api_process
    for name, proc in [('price updater', updater_process), ('portfolio GUI', portfolio_process), ('portfolio API', portfolio_api_process)]:
        if proc:
            try:
                print(f"\n[INFO] Closing {name}...")
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


# Register cleanup on exit
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def _start_updater(script_dir):
    """Start the price updater in the background. Returns the process or None."""
    print("\n[INFO] Starting price updater...")
    try:
        log_path = os.path.join(script_dir, 'updater.log')
        with open(log_path, 'w') as log_file:
            log_file.write(f"=== Price Updater Started {datetime.datetime.now().isoformat()} ===\n")
            log_file.flush()
            proc = subprocess.Popen(
                [sys.executable, 'GE Price Update.py'],
                cwd=script_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        print(f"  Price updater started (PID: {proc.pid})")
        time.sleep(2)
        return proc
    except Exception as e:
        print(f"  Warning: Could not start price updater: {e}")
        return None


def _start_portfolio_api(script_dir):
    """Start the RuneLite portfolio API on port 8052. Returns the process or None."""
    print("\n[INFO] Starting portfolio API on http://localhost:8052 ...")
    try:
        log_path = os.path.join(script_dir, 'portfolio_api.log')
        with open(log_path, 'w') as log_file:
            log_file.write(f"=== Portfolio API Started {datetime.datetime.now().isoformat()} ===\n")
            log_file.flush()
            proc = subprocess.Popen(
                [sys.executable, 'portfolio_api.py'],
                cwd=script_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        print(f"  Portfolio API started (PID: {proc.pid})")
        time.sleep(1)
        return proc
    except Exception as e:
        print(f"  Warning: Could not start portfolio API: {e}")
        return None


def _start_portfolio_gui(script_dir):
    """Start the portfolio GUI in the background. Returns the process or None."""
    print("\n[INFO] Starting portfolio manager on http://localhost:8051 ...")
    try:
        proc = subprocess.Popen(
            [sys.executable, 'portfolio_gui.py'],
            cwd=script_dir,
        )
        print(f"  Portfolio GUI started (PID: {proc.pid})")
        time.sleep(1)
        return proc
    except Exception as e:
        print(f"  Warning: Could not start portfolio GUI: {e}")
        return None


def main():
    global updater_process, portfolio_process, portfolio_api_process

    args = sys.argv[1:]
    run_main = '--portfolio' not in args          # default: run main GUI
    run_portfolio = '--portfolio' in args or '--both' in args
    if '--both' in args:
        run_main = True

    script_dir = (
        os.path.dirname(os.path.abspath(__file__))
        if not getattr(sys, 'frozen', False)
        else os.path.dirname(sys.executable)
    )
    print(f"Script directory: {script_dir}")

    # Always start price updater and portfolio API (RuneLite plugin needs port 8052)
    updater_process = _start_updater(script_dir)
    portfolio_api_process = _start_portfolio_api(script_dir)

    # Optionally start portfolio in background
    if run_portfolio and not run_main:
        # Portfolio-only mode: run in foreground so the terminal stays alive
        print("\n[INFO] Portfolio-only mode. Open http://localhost:8051 in your browser.")
        try:
            subprocess.run([sys.executable, 'portfolio_gui.py'], cwd=script_dir)
        except Exception as e:
            print(f"ERROR: {e}")
        finally:
            cleanup()
            print("[INFO] Goodbye!")
        return

    if run_portfolio and run_main:
        portfolio_process = _start_portfolio_gui(script_dir)

    # Run main GUI (blocking)
    gui_exe = os.path.join(script_dir, 'ge_gui.exe')
    print(f"\n[INFO] Starting main GUI application...")
    try:
        if getattr(sys, 'frozen', False) and os.path.exists(gui_exe):
            print(f"  Running from: {gui_exe}")
            subprocess.run([gui_exe], cwd=script_dir)
        else:
            print("  Running Python script")
            subprocess.run([sys.executable, 'analyze_advanced_gui.py'], cwd=script_dir)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()
        print("[INFO] Goodbye!")


if __name__ == '__main__':
    main()
