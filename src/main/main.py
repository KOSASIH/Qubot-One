from flask import Flask
from config import Config
from logger import setup_logging
from event_loop import EventLoop
from task_scheduler import TaskScheduler
from user_interface import UserInterface
from cloud_integration import CloudIntegration

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Setup logging
    setup_logging()

    # Initialize components
    event_loop = EventLoop()
    task_scheduler = TaskScheduler()
    user_interface = UserInterface()
    cloud_integration = CloudIntegration()

    return app, event_loop, task_scheduler, user_interface, cloud_integration

if __name__ == "__main__":
    app, event_loop, task_scheduler, user_interface, cloud_integration = create_app()
    app.run(host='0.0.0.0', port=5000)
