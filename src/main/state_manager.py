class StateManager:
    def __init__(self):
        self.state = {}

    def set_state(self, qubot_id, state):
        self.state[qubot_id] = state

    def get_state(self, qubot_id):
        return self.state.get(qubot_id, None)

    def reset_state(self, qubot_id):
        if qubot_id in self.state:
            del self.state[qubot_id]
