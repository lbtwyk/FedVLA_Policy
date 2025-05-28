import flwr as fl
import torch
import logging

# Import the ClientApp from your flwr_client.py
# Make sure flwr_client.py is in the same directory or Python path
from flwr_client import app as client_app 
# Import the ServerApp from your flwr_server.py
from flwr_server import app as server_app, NUM_ROUNDS as server_default_num_rounds

# Configure basic logging for the simulation script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - SIM - %(levelname)s - %(message)s')

# --- Simulation Configuration ---
NUM_CLIENTS_TO_SIMULATE = 3 # Matches the NUM_TOTAL_CLIENTS in flwr_client.py
NUM_SIMULATION_ROUNDS = server_default_num_rounds # Use server default or override

# It's important that NUM_CLIENTS_TO_SIMULATE matches how NUM_TOTAL_CLIENTS
# is used in flwr_client.py for data partitioning and in flwr_server.py for min_clients settings.

if __name__ == "__main__":
    logging.info(f"Starting Flower simulation with {NUM_CLIENTS_TO_SIMULATE} clients for {NUM_SIMULATION_ROUNDS} rounds.")

    # Define backend configuration (resources for clients)
    # By default, Flower assigns 2 CPUs per client if client_resources is None.
    # If running on a machine with a GPU and PyTorch is MPS or CUDA enabled:
    if torch.backends.mps.is_available() or torch.cuda.is_available():
        # This tells Flower that each client *could* use a GPU if it asks for one.
        # The client-side code (train_client_model, test_client_model) will automatically use MPS/CUDA if available.
        # For actual GPU assignment in complex setups, more detailed resource management might be needed.
        # For local simulation, this is often conceptual unless using Ray with specific resource requests.
        # Flower's default virtual client execution handles device placement within the client code itself.
        backend_config = {"client_resources": {"num_gpus": 0, "num_cpus": 2}} # Default, GPU use is handled by client code
        logging.info("MPS or CUDA available. Client code will attempt to use GPU if available.")
        # To request GPU for each client from Ray (if using Ray backend, not default): 
        # backend_config = {"client_resources": {"num_gpus": 1, "num_cpus": 1}} 
    else:
        backend_config = {"client_resources": {"num_gpus": 0, "num_cpus": 2}}
        logging.info("No MPS or CUDA available. Clients will use CPU.")

    # Start the simulation
    history = fl.simulation.run_simulation(
        server_app=server_app,       # Your ServerApp instance
        client_app=client_app,       # Your ClientApp instance
        num_supernodes=NUM_CLIENTS_TO_SIMULATE, # Number of clients to simulate
        backend_config=backend_config,
    )

    logging.info("Simulation finished.")
    logging.info(f"History: {history}")

    # You can add code here to plot metrics from the history object, e.g.:
    # import matplotlib.pyplot as plt
    # print(f"Losses distributed: {history.losses_distributed}")
    # print(f"Metrics distributed (accuracy): {history.metrics_distributed['accuracy']}")
    # print(f"Metrics centralized: {history.metrics_centralized}") # Will be empty with FedAvg

    # Example plotting (requires metrics to be properly logged by strategy)
    # if history.losses_distributed:
    #     rounds, losses = zip(*history.losses_distributed)
    #     plt.plot(rounds, losses)
    #     plt.xlabel("Round")
    #     plt.ylabel("Distributed Loss")
    #     plt.title("Federated Learning Loss Over Rounds")
    #     plt.show()

    # if 'accuracy' in history.metrics_distributed and history.metrics_distributed['accuracy']:
    #     rounds, accuracies = zip(*history.metrics_distributed['accuracy'])
    #     plt.plot(rounds, accuracies)
    #     plt.xlabel("Round")
    #     plt.ylabel("Distributed Accuracy")
    #     plt.title("Federated Learning Accuracy Over Rounds")
    #     plt.show() 