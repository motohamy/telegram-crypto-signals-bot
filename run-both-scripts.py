import threading
import subprocess
import sys
import time
import os
#  AIzaSyDK0JbwANnhWfqK2HTkHNvRvjD3mBVY6ew
def run_webhook_receiver():
    """Run the webhook receiver script"""
    print("Starting webhook receiver...")
    try:
        subprocess.run([sys.executable, "C:\\path\\alex\\ai-agent\\predictor\\webhook-receiver.py"])
    except Exception as e:
        print(f"Error running webhook receiver: {e}")

def run_trading_script():
    """Run the crypto trading script"""
    print("Starting crypto trading script...")
    try:
        # Give webhook receiver a moment to start up
        time.sleep(3)
        subprocess.run([sys.executable, "C:\\path\\alex\\ai-agent\\predictor\\enhanced-bot-complete.py"])
    except Exception as e:
        print(f"Error running trading script: {e}")

if __name__ == "__main__":
    # Make sure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Create and start webhook receiver thread
    webhook_thread = threading.Thread(target=run_webhook_receiver)
    webhook_thread.daemon = True  # This makes the thread exit when the main program exits
    webhook_thread.start()
    
    # Run the trading script in the main thread
    run_trading_script()
    
    # Keep the program running until Ctrl+C is pressed
    try:
        while webhook_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
