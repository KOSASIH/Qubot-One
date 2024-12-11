import sched
import time

class TaskScheduler:
    def __init__(self):
        self.scheduler = sched.scheduler(time.time, time.sleep)

    def schedule_task(self, delay, task, *args):
        self.scheduler.enter(delay, 1, task, args)

    def run_pending(self):
        self.scheduler.run()
