class MultiAgentSystem:
    def __init__(self):
        self.agents = []

    def add_agent(self, agent):
        self.agents.append(agent)

    def coordinate_agents(self):
        for agent in self.agents:
            agent.perform_action()
