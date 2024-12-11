import asyncio

class EventLoop:
    def __init__(self):
        self.loop = asyncio.get_event_loop()

    def run(self):
        try:
            self.loop.run_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.loop.close()

    def create_task(self, coro):
        return self.loop.create_task(coro)
