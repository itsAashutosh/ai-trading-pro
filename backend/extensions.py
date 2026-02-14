from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from apscheduler.schedulers.background import BackgroundScheduler

db = SQLAlchemy()
migrate = Migrate()
# Scheduler will be initialized here but we must be careful not to start it twice
# if running alongside the legacy app in the same process (which isn't likely how we run it, but still)
scheduler = BackgroundScheduler()
