class SimulationManager:
    def __init__(self):
        self.simulations = []

    def start_simulation(self, simulation):
        self.simulations.append(simulation)
        simulation.start()

    def stop_simulation(self, simulation):
        simulation.stop()
        self.simulations.remove(simulation)
