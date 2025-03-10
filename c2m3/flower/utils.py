from flwr.server.strategy import FedAvg

class CustomStrategy(FedAvg):
    def __init__(self, initial_parameters_list):
        super().__init__()
        self.initial_parameters_list = initial_parameters_list
        self.client_index = 0

    def initialize_parameters(self, client_manager):
        # Logic to cycle through or choose parameter sets for each client
        params = self.initial_parameters_list[self.client_index]
        self.client_index = (self.client_index + 1) % len(self.initial_parameters_list)
        return params
