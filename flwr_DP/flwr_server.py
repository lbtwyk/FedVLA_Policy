import flwr as fl
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
from typing import List, Tuple, Optional, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default number of rounds, can be overridden by run_simulation.py or context
NUM_ROUNDS = 500

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics using weighted average."""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    aggregated_metrics: Dict[str, float] = {}
    if total_examples == 0:
        logging.warning("weighted_average received metrics with 0 total_examples.")
        return {"accuracy": 0.0, "eval_loss": 0.0} 

    for num_examples, client_metrics in metrics:
        for metric_name, metric_value in client_metrics.items():
            current_metric_value = aggregated_metrics.get(metric_name, 0.0)
            # Ensure metric_value is float before multiplication
            aggregated_metrics[metric_name] = current_metric_value + (float(metric_value) * num_examples / total_examples)
    return aggregated_metrics

class CustomFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_round = 0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate model weights and log training progress."""
        self.current_round = server_round
        logging.info(f"\nRound {server_round} - Aggregating fit results from {len(results)} clients")
        
        aggregated_loss = 0.0
        total_examples_fit = 0

        for client_proxy, fit_res in results:
            if fit_res.metrics and 'train_loss' in fit_res.metrics:
                logging.info(f"Client {client_proxy.cid} - Training loss: {fit_res.metrics['train_loss']:.4f}, Num Examples: {fit_res.num_examples}")
                # Ensure train_loss is float
                aggregated_loss += float(fit_res.metrics['train_loss']) * fit_res.num_examples
                total_examples_fit += fit_res.num_examples
            else:
                logging.info(f"Client {client_proxy.cid} - FitRes metrics: {fit_res.metrics} (train_loss not found or no metrics)")
        
        aggregated_parameters_tuple = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters_tuple is None:
            logging.warning(f"Round {server_round} - Failed to aggregate model weights (super().aggregate_fit returned None)")
            return None, {}

        aggregated_parameters, _ = aggregated_parameters_tuple 

        if aggregated_parameters is not None:
            logging.info(f"Round {server_round} - Successfully aggregated model weights")
        else:
            logging.warning(f"Round {server_round} - Failed to aggregate model weights")

        fit_metrics = {}
        if total_examples_fit > 0:
            fit_metrics["avg_train_loss"] = aggregated_loss / total_examples_fit
        else:
            fit_metrics["avg_train_loss"] = float('inf') # Or 0.0, or skip if no examples
        
        return aggregated_parameters, fit_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        """Aggregate evaluation results and log progress."""
        logging.info(f"\nRound {server_round} - Aggregating evaluation results from {len(results)} clients")
        
        if not results:
            logging.warning(f"Round {server_round} - No evaluation results to aggregate.")
            return None, {}

        all_client_metrics = []
        for client_proxy, eval_res in results:
            if eval_res.metrics:
                logging.info(
                    f"Client {client_proxy.cid} - "
                    f"Loss: {eval_res.loss:.4f}, " 
                    f"Metrics: {eval_res.metrics}, "
                    f"Samples: {eval_res.num_examples}"
                )
                if eval_res.num_examples > 0:
                    all_client_metrics.append((eval_res.num_examples, eval_res.metrics))
                else:
                    logging.warning(f"Client {client_proxy.cid} reported 0 examples in evaluation. Skipping its metrics for weighted average.")
            else:
                logging.warning(f"Client {client_proxy.cid} - EvaluateRes had no metrics. Loss: {eval_res.loss}")

        if not all_client_metrics:
            logging.warning(f"Round {server_round} - No valid client metrics to aggregate for evaluation.")
            loss_aggregated, _ = super().aggregate_evaluate(server_round, results, failures)
            return loss_aggregated, {}

        aggregated_metrics = weighted_average(all_client_metrics)
        loss_aggregated, _ = super().aggregate_evaluate(server_round, results, failures)

        logging.info(
            f"Round {server_round} - Aggregated Evaluation Results: Loss: {loss_aggregated if loss_aggregated is not None else 'N/A'}, Metrics: {aggregated_metrics}"
        )
        # Ensure loss_aggregated is float if not None, else keep None
        final_loss_aggregated = float(loss_aggregated) if loss_aggregated is not None else None
        return final_loss_aggregated, aggregated_metrics

# Define a function that creates the ServerApp components
def server_fn(context: Context) -> ServerAppComponents:
    """Return the ServerApp components."""
    # Get num_total_clients from the context if available, else use a default
    # This value is passed from run_simulation.py via server_app_config
    num_total_clients_for_strategy = context.run_config.get("num_total_clients_for_strategy", 3) # Default to 3 if not in context
    logging.info(f"Server_fn: Using {num_total_clients_for_strategy} for min_fit/eval/available clients in strategy.")

    strategy = CustomFedAvg(
        fraction_fit=1.0,  
        fraction_evaluate=1.0,  
        min_fit_clients=num_total_clients_for_strategy,  
        min_evaluate_clients=num_total_clients_for_strategy, 
        min_available_clients=num_total_clients_for_strategy, 
        evaluate_metrics_aggregation_fn=weighted_average, 
    )

    # Get num_rounds from context if available, else use module default
    num_rounds_for_server = context.run_config.get("num_rounds", NUM_ROUNDS)
    logging.info(f"Server_fn: Configuring server for {num_rounds_for_server} rounds.")
    config = ServerConfig(num_rounds=num_rounds_for_server) 

    return ServerAppComponents(
        strategy=strategy, 
        config=config
    )

# Create ServerApp instance
app = ServerApp(
    server_fn=server_fn,
)

# No main function here anymore, it will be in run_simulation.py
# Old main() function removed.
