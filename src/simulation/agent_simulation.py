# src/simulation/agent_simulation.py

class AgentSimulation:
    def __init__(self, environment):
        """Initialize the agent simulation.

        Args:
            environment (SimulationEnvironment): The environment for the agents.
        """
        self.environment = environment
        self.agents = []

    def add_agent(self, agent):
        """Add an agent to the simulation.

        Args:
            agent (object): The agent to add.
        """
        self.agents.append(agent)
        self.environment.add_entity(agent)
        print(f"Agent {agent} added to the simulation.")

    def update_agents(self, delta_time):
        """Update the state of all agents.

        Args:
            delta_time (float): Time step for the update.
        """
        for agent in self.agents:
            if hasattr(agent, 'update'):
                agent.update(delta_time)
                print(f"Updated agent {agent}.")
