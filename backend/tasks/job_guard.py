import time
import functools
import threading
import uuid
from datetime import datetime, timedelta
from flask import current_app

class JobRegistry:
    _active_jobs = {}
    _lock = threading.Lock()
    
    @classmethod
    def register(cls, job_name, job_id=None):
        job_id = job_id or str(uuid.uuid4())
        with cls._lock:
            cls._active_jobs[job_id] = {
                'name': job_name,
                'started_at': datetime.utcnow(),
                'thread_id': threading.get_ident()
            }
        return job_id
        
    @classmethod
    def unregister(cls, job_id):
        with cls._lock:
            if job_id in cls._active_jobs:
                del cls._active_jobs[job_id]

    @classmethod
    def get_stuck_jobs(cls, timeout_minutes=60):
        now = datetime.utcnow()
        limit = timedelta(minutes=timeout_minutes)
        with cls._lock:
            return [
                (jid, info) for jid, info in cls._active_jobs.items()
                if now - info['started_at'] > limit
            ]

def monitor_job(name, timeout_minutes=60):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            job_id = JobRegistry.register(name)
            start_time = time.time()
            
            # Logger check (handle uninitialized app context)
            logger = None
            try:
                if current_app:
                    logger = current_app.logger
            except:
                pass
                
            if logger:
                logger.info(f"START JOB: {name} (ID: {job_id})")
                
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                if logger:
                    logger.info(f"END JOB: {name} | Duration: {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                if logger:
                    logger.error(f"JOB FAILED: {name} | Error: {str(e)}", extra={'stack_trace': True})
                raise e
            finally:
                JobRegistry.unregister(job_id)
        return wrapper
    return decorator
