import warnings
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server.strategy.dpfedavg_fixed import DPFedAvgFixed
from flwr.client.numpy_client import NumPyClient
from flwr.client.dpfedavg_numpy_client import DPFedAvgNumPyClient
import numpy as np
from sklearn.metrics import log_loss, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from typing import Callable, Optional, Tuple, List, Dict
from flwr.common.typing import Scalar, NDArrays
import pickle
import utilsSim
import json
import pandas as pd
import random
import os
import argparse

parser = argparse.ArgumentParser(description='Train a model in federated learning')
parser.add_argument(
  "--model_version",
  dest='model_version',
  help="Version of the model",
  required=True
)
parser.add_argument(
  "--output_dir",
  dest='output_dir',
  help="output directory",
  required=True
)
parser.add_argument(
  "--num_clients",
  dest='num_clients',
  help="The number of clients for training",
  type= int,
  default= 10000
)
parser.add_argument(
  "--num_rounds",
  dest='num_rounds',
  help="The number of rounds of training",
  type= int,
  default= 100
)

parser.add_argument(
  "--fraction_fit",
  dest = "fraction_fit",
  help = "how many client i want to select for training at each round ",
  type = float,
  default = 0.01
)

parser.add_argument(
  "--num_sampled_clients",
  dest = "num_sampled_clients",
  help = "number of sampled clients",
  type = int,
  default=100
)

#Privacy parameters
parser.add_argument(
  "--clip_norm",
  dest = "clip_norm",
  help = "clipping threshold: the influence of each client’s update is bounded by clipping it. This is achieved by enforcing a cap on the L2 norm of the update, scaling it down if needed.",
  type = float,
  default = 0.10
)

parser.add_argument(
  "--noise_multiplier",
  dest = "noise_multiplier",
  help = "Amount of gaussian noise to add",
  type = float,
  default = 1.4
)

parser.add_argument(
  "--server_side_noising",
  dest='server_side_noising',
  help="True or False: Add noise at the client level or at the server level",
  default=True
)

args = parser.parse_args()


# Define Flower client
class FlowerClient(fl.client.dpfedavg_numpy_client.DPFedAvgNumPyClient):
    def __init__(self, cid, client_model, X_train, y_train, X_val, y_val) -> None:
        self.cid = cid
        self.client_model = client_model
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val

    def get_parameters(self, config):  # type: ignore
        return utilsSim.get_model_parameters(self.client_model)

    def fit(self, parameters, config):  # type: ignore
        utilsSim.set_model_params(self.client_model, parameters)
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.client_model.fit(self.X_train, self.y_train)
        print(f"Training finished for client {self.cid}, for round {config['rnd']}")

        return utilsSim.get_model_parameters(self.client_model), len(self.X_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        utilsSim.set_model_params(self.client_model, parameters)
        loss = log_loss(self.y_val, self.client_model.predict_proba(self.X_val))
        accuracy = self.client_model.score(self.X_val, self.y_val)
        y_pred = self.client_model.predict(self.X_val)
        f1 = f1_score(self.y_val, y_pred)
        precision = precision_score(self.y_val, y_pred)
        recall = recall_score(self.y_val, y_pred)

        return loss, len(self.X_val), {"accuracy": accuracy, "f1": f1, "precision" : precision, "recall" : recall}


def client_fn(cid: str) -> fl.client.Client:
    #Create model
    client_model = LogisticRegression(
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

    #Set initial parameters
    utilsSim.set_initial_params(client_model)

    #Load the data
    # Load the different datasets: warmup, training and validation
    #Load the embeddings+labels
#Load the embeddings+labels
    with open('all_chatnames.pkl', 'rb') as c: #Add path to file
        all_names = pickle.load(c)
    with open('all_embeddings.pkl', 'rb') as e:
        all_embeddings = pickle.load(e)    
    with open('all_labels.pkl', 'rb') as l:
        all_labels = pickle.load(l)
    #Load the users list
    with open('val_list.pkl', 'rb') as v:
        val_list = pickle.load(v)
    with open('train_list.pkl', 'rb') as t:
        train_list = pickle.load(t)
    with open('warmup_list.pkl', 'rb') as b:
        warmup_list = pickle.load(b)
    with open('train_nonpred_list.pkl', 'rb') as z:
        train_nonpred_list = pickle.load(z)



    #Create pandas dataframe
    df = pd.DataFrame(list(zip(all_names, all_embeddings, all_labels)), columns=['chatNames','embeddings','labels'])


    name_list=[]
    for index, row in df.iterrows():
        if (row['labels']==1) :
            df.at[index,'names'] = row['chatNames'].rsplit('-',1)[0]
        else:
            df.at[index,'names'] = row['chatNames']
            
    #Create warmup data: 1 row positive and 1 row negative
    df_warmup_all = df[df['names'].isin(warmup_list)]
    df_warmup_pred = df_warmup_all.query('(labels==1)').sample(n=10)
    df_warmup_nonpred = df_warmup_all.query('(labels==0)').sample(n=10)

    #Select a user randomly from the train set
    user = random.choice(train_list)
    # user = train_list[int(cid)]
    df_user = df[df['names'].str.contains(user)]

    if df_user['labels'].values.any() == 0:
        add_neg = random.choices(train_nonpred_list, k=10)
        df_neg = df[df['names'].isin(add_neg)]
        df_client = pd.concat([df_user, df_neg])
    else:
        df_client = df_user

 

    #Load the training set = a base with both labels + one user chosen randomly for each client
    X_train = pd.concat([df_warmup_pred, df_warmup_nonpred, df_client])['embeddings'].to_list() #Du coup ca ca va etre les exemples dans base + la conversation choisie aléatoirement
    y_train = pd.concat([df_warmup_pred, df_warmup_nonpred, df_client])['labels'].to_list()#Du coup ca ca va etre les exemples dans base + la conversation choisie aléatoirement

    #Load the validation set
    X_val = df[df['names'].isin(val_list)]['embeddings'].to_list()
    y_val = df[df['names'].isin(val_list)]['labels'].to_list()

    #Create and return client
    return FlowerClient(cid, client_model, X_train, y_train, X_val, y_val)


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"rnd": server_round}
    
   
def get_eval_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    #Load the embeddings+labels
    with open('all_chatnames.pkl', 'rb') as c:
        all_names = pickle.load(c)
    with open('all_embeddings.pkl', 'rb') as e:
        all_embeddings = pickle.load(e)    
    with open('all_labels.pkl', 'rb') as l:
        all_labels = pickle.load(l)



    #Create pandas dataframe
    df = pd.DataFrame(list(zip(all_names, all_embeddings, all_labels)), columns=['chatNames','embeddings','labels'])

    name_list=[]
    for index, row in df.iterrows():
        if (row['labels']==1) :
            df.at[index,'names'] = row['chatNames'].rsplit('-',1)[0]
        else:
            df.at[index,'names'] = row['chatNames']



    #Load lists
    with open('val_list.pkl', 'rb') as v:
        val_list = pickle.load(v)

    #Load the validation set
    X_val = df[df['names'].isin(val_list)]['embeddings'].to_list()
    y_val = df[df['names'].isin(val_list)]['labels'].to_list()


    # The `evaluate` function will be called after every round
    def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # Update model with the latest parameters
        utilsSim.set_model_params(model, parameters)
        loss = log_loss(y_val, model.predict_proba(X_val))
        accuracy = model.score(X_val, y_val)
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val,y_pred)
        precision = precision_score(y_val,y_pred)
        recall = recall_score(y_val,y_pred)
        name_model = 'model-%s-c%s-ff%s-r%s-sc%s-cn%s-nm%s.sav' % (args.model_version, args.num_clients, args.fraction_fit, args.num_rounds, args.num_sampled_clients, args.clip_norm, args.noise_multiplier)
        model_file = os.path.join(args.output_dir, name_model)
        pickle.dump(model, open(model_file, 'wb'))
        name_results = 'results-%s-c%s-ff%s-r%s-sc%s-cn%s-nm%s.txt' % (args.model_version, args.num_clients, args.fraction_fit, args.num_rounds,  args.num_sampled_clients, args.clip_norm, args.noise_multiplier)
        results_file = os.path.join(args.output_dir, name_results)
        with open (results_file, 'a') as results:
          results.write(json.dumps({"loss": loss, "accuracy": accuracy, "f1": f1, "precision" : precision, "recall" : recall}) + "\n")

        return loss, {"accuracy": accuracy, "f1": f1, "precision" : precision, "recall" : recall}


    return evaluate


server_model = LogisticRegression()
utilsSim.set_initial_params(server_model)




# Create DP-FedAvg strategy
strategydp= fl.server.strategy.dpfedavg_fixed.DPFedAvgFixed( 
  strategy = fl.server.strategy.FedAvg(
      fraction_fit=args.fraction_fit,  # Sample 10% of available clients for training
      fraction_evaluate=0.10,  # Sample 5% of available clients for evaluation
      # min_fit_clients=10,  # Never sample less than 10 clients for training
      # min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
      min_available_clients=int(args.num_clients*0.10),  # Wait until at least 75% clients are available
      evaluate_fn=get_eval_fn(server_model),
      on_fit_config_fn=fit_round,
    ),
  num_sampled_clients=args.num_sampled_clients,
  clip_norm=args.clip_norm,
  noise_multiplier=args.noise_multiplier,
  server_side_noising=args.server_side_noising
)

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=args.num_clients,
    config=fl.server.ServerConfig(num_rounds=args.num_rounds),
    strategy=strategydp
)










