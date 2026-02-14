import signal
import sys
import time
from backend.extensions import scheduler
from backend.tasks.job_guard import JobRegistry

def graceful_shutdown(signum, frame):
    """
    Handle termination signals to perform graceful shutdown.
    """
    print("\n\n🛑 Received shutdown signal. Initiating graceful shutdown...")
    
    # 1. Stop Scheduler
    if scheduler.running:
        print("⏳ Stopping scheduler (waiting for scheduled jobs)...")
        scheduler.shutdown(wait=True)
        print("✅ Scheduler stopped.")
        
    # 2. Wait for Background Jobs (RL Training, etc.)
    active_jobs = JobRegistry.get_stuck_jobs(timeout_minutes=0) # Get all jobs
    count = len(active_jobs)
    
    if count > 0:
        print(f"⏳ Waiting for {count} active background jobs to complete...")
        # Simple wait loop
        while len(JobRegistry.get_stuck_jobs(timeout_minutes=0)) > 0:
            sys.stdout.write(".")
            sys.stdout.flush()
            time.sleep(1)
        print("\n✅ All jobs completed.")
    else:
        print("✅ No active background jobs.")
        
    print("👋 Bye!")
    sys.exit(0)

def register_signal_handlers(app):
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)
