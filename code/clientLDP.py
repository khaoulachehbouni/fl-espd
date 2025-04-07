from collections import OrderedDict
from typing import Callable, Optional, Tuple, List, Dict
import flwr as fl
from flwr.common.typing import Scalar
import torch
from utilsLDP import load_data, LogisticRegression, LRClient, test, get_params, set_params
import numpy as np
import json
import os
import argparse
import pickle

parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")


#Simulation parameters
parser.add_argument("--model_version", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--num_clients", type=int, default=10000)
parser.add_argument("--num_rounds", type=int, default=100)

#Privacy parameters
parser.add_argument("--learning_rate", type=float, default=0.05)    
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--noisy_embeddings", required=True)


args = parser.parse_args()



# Define an evaluation function for centralized evaluation
def get_evaluate_fn() -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, float]]:
        
        trainloader, valloader = load_data(args.noisy_embeddings)
        model = LogisticRegression(768, 1)
        #Set model parameters
              # Set weights in model.
        set_params(model, parameters)
        loss, accuracy, f1, precision, recall = test(model, valloader)
        name_model = 'model-%s-c%s-r%s-lr%s-e%s-eps%s.pth' % (args.model_version, args.num_clients, args.num_rounds, args.learning_rate, args.epochs, args.noisy_embeddings)
        model_file = os.path.join(args.output_dir, name_model)
        torch.save(model.state_dict(), model_file)
        name_results = 'results-%s-c%s-r%s-lr%s-e%s-eps%s.txt' % (args.model_version, args.num_clients, args.num_rounds, args.learning_rate, args.epochs, args.noisy_embeddings)
        results_file = os.path.join(args.output_dir, name_results)
        with open (results_file, 'a') as results:
          results.write(json.dumps({"loss": loss, "accuracy": accuracy, "f1": f1, "precision" : precision, "recall" : recall}) + "\n")
        # Return metrics.
        return loss, {"accuracy": accuracy, "f1": f1, "precision" : precision, "recall" : recall}

    return evaluate


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": args.epochs  # number of local epochs
    }
    return config


if __name__ == "__main__":
  # parse input arguments
  args = parser.parse_args()

  def client_fn(cid: str) -> fl.client.Client:
    model = LogisticRegression(768, 1)
    #Load data 
    trainloader, valloader = load_data(args.noisy_embeddings)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    return LRClient(model, trainloader, valloader, optimizer)

  # configure the strategy
  strategy = fl.server.strategy.FedAvg(
      fraction_fit=0.1,
      fraction_evaluate=0.1,
      # min_fit_clients=10,
      # min_evaluate_clients=10,
      # min_available_clients=pool_size,  
      on_fit_config_fn=fit_config,
      evaluate_fn=get_evaluate_fn(),  # centralised evaluation of global model
  )

  # (optional) specify Ray config
#   ray_init_args = {"include_dashboard": False}

  # Start Flower simulation
  fl.simulation.start_simulation(
      client_fn=client_fn,
      num_clients=args.num_clients,
      config=fl.server.ServerConfig(num_rounds=args.num_rounds),
      strategy=strategy#,
    #   ray_init_args=ray_init_args
  )

