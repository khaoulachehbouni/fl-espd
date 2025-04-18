import pandas as pd
import numpy as np
import torch
import pickle
import flwr as fl
import random
from torch.utils.data import DataLoader
import tqdm
from collections import OrderedDict
from typing import Callable, Optional, Tuple, List, Dict
import warnings
import os
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
warnings.simplefilter("ignore")

BATCH_SIZE = 100

#Define model used for training
class LogisticRegression(torch.nn.Module):
  def __init__(self, input_dim, output_dim):
      super(LogisticRegression, self).__init__()
      self.linear = torch.nn.Linear(input_dim, output_dim)

  def forward(self, x):
      outputs = torch.sigmoid(self.linear(x))
      return outputs

def accuracy(preds, labels):
  return (preds == labels).mean()


def train(model, trainloader, optimizer, epoch):
  #Instantiate the Loss Class
  criterion = torch.nn.BCELoss() # computes softmax and then the cross entropy
  optimizer = optimizer 
  losses = []
  top1_acc = []
  
  for i, (embeddings, labels) in enumerate(trainloader):  
    optimizer.zero_grad()
    # compute output
    output = torch.squeeze(model(embeddings))
    loss = criterion(output, labels) #Returns a tensor with all the dimensions of input of size 1 removed.
    loss.backward()
    optimizer.step()
    
    # measure accuracy and record loss
    if i%20==0:
      correct_preds = 0
      total_preds = 0
      
      total_preds+= labels.size(0)

      preds = output.round().detach().numpy()
      labels = labels.detach().numpy()
      
      correct_preds += np.sum(preds == labels)
      acc = correct_preds/total_preds
      
      losses.append(loss.item())
      top1_acc.append(acc)
      # epsilon = privacy_engine.get_epsilon(delta)
      print(
          f"\tTrain Epoch: {epoch} \t"
          f"Loss: {np.mean(losses):.6f} "
          f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
          # f"(ε = {epsilon:.2f}, δ = {delta})"
        )
      


def test(model, valloader):
  criterion = torch.nn.BCELoss()
  losses = []
  preds_all = []
  labels_all = []

  with torch.no_grad():
    for embeddings, labels in valloader:
        output = torch.squeeze(model(embeddings))
        loss = criterion(output, labels)
        
        preds = output.round().detach().numpy().tolist()
        labels = labels.detach().numpy().tolist()

        # correct_preds += np.sum(preds == labels)
        

        losses.append(loss.item())

        labels_all.extend(labels)

        preds_all.extend(preds)


        # top1_acc.append(acc)

#   top1_avg = np.mean(top1_acc)


  acc = accuracy_score(labels_all, preds_all)
  f1 = f1_score(labels_all, preds_all)
  precision = precision_score(labels_all, preds_all)
  recall = recall_score(labels_all, preds_all)
  loss = np.mean(losses)
  return loss, acc, f1, precision, recall


def get_params(model: torch.nn.ModuleList) -> List[np.ndarray]:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.detach().numpy() for _, val in model.state_dict().items()]

def set_params(model: torch.nn.ModuleList, params: List[np.ndarray]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


#Define Flower Client
class LRClient(fl.client.NumPyClient):
  def __init__(self, model, trainloader, valloader, optimizer) -> None:
    self.model = model
    self.trainloader = trainloader
    self.valloader = valloader
    self.optimizer = optimizer


  def get_parameters(self, config):
      return get_params(self.model)

  def fit(self, parameters, config):
      set_params(self.model, parameters)
      for epoch in range(config["epochs"]):
        train(self.model, self.trainloader, self.optimizer, epoch + 1)
      
      return get_params(self.model), len(self.trainloader), {} 

  def evaluate(self, parameters, config):
      set_params(self.model, parameters)
      loss, acc, f1, precision, recall = test(self.model, self.valloader)
      return float(loss), len(self.valloader), {"accuracy": acc, "f1": f1, "precision" : precision, "recall" : recall}




def load_data(noisy_embeddings):
  #Load the embeddings+labels
    with open('all_chatnames.pkl', 'rb') as c:#Add path to file
        all_names = pickle.load(c)
    path_to_file = "data" #Add path to file
    embeddings = os.path.join(path_to_file, noisy_embeddings)
    with open(embeddings, 'rb') as e:
        all_embeddings = pickle.load(e)
    noisy_embeddings_wnoise = "all_embeddings.pkl"
    embeddings_wnoise = os.path.join(path_to_file, noisy_embeddings_wnoise)
    with open(embeddings_wnoise, 'rb') as j:    
        all_embeddings_wnoise = pickle.load(j)
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
    df_val = pd.DataFrame(list(zip(all_names, all_embeddings_wnoise, all_labels)), columns=['chatNames','embeddings','labels'])

    name_list=[]
    for index, row in df.iterrows():
        if (row['labels']==1) :
            df.at[index,'names'] = row['chatNames'].rsplit('-',1)[0]
        else:
            df.at[index,'names'] = row['chatNames']
    
    for index, row in df_val.iterrows():
        if (row['labels']==1) :
            df_val.at[index,'names'] = row['chatNames'].rsplit('-',1)[0]
        else:
            df_val.at[index,'names'] = row['chatNames']


    #Create warmup data: 1 row positive and 1 row negative
    df_warmup_all = df[df['names'].isin(warmup_list)]
    df_warmup_pred = df_warmup_all.query('(labels==1)').sample(n=10)
    df_warmup_nonpred = df_warmup_all.query('(labels==0)').sample(n=10)

    #Select a user randomly from the train set
    user = random.choice(train_list)
    # user = train_list[int(cid)]
    df_user = df[df['names'].str.contains(user)]

    if df_user['labels'].values.all() == 0:
        add_neg = random.choices(train_nonpred_list, k=10)
        df_neg = df[df['names'].isin(add_neg)]
        df_client = pd.concat([df_user, df_neg])
    else:
        df_client = df_user

    #Load the training set = a base with both labels + one user chosen randomly for each client
    X_train = pd.concat([df_warmup_pred, df_warmup_nonpred, df_client])['embeddings'].to_list() #Du coup ca ca va etre les exemples dans base + la conversation choisie aléatoirement
    y_train = pd.concat([df_warmup_pred, df_warmup_nonpred, df_client])['labels'].to_list()#Du coup ca ca va etre les exemples dans base + la conversation choisie aléatoirement

    #Load the validation set
    X_val = df_val[df_val['names'].isin(val_list)]['embeddings'].to_list()
    y_val = df_val[df_val['names'].isin(val_list)]['labels'].to_list()


    # Create torch dataset
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, embeddings, labels):
            self.embeddings = embeddings
            self.labels = labels

        def __getitem__(self, idx):
            embeddings = torch.tensor(self.embeddings[idx], dtype=torch.float32)
            labels = torch.tensor(self.labels[idx], dtype=torch.float32)
            return embeddings, labels

        def __len__(self):
            return len(self.embeddings)

    train_dataset = Dataset(X_train, y_train)
    val_dataset = Dataset(X_val, y_val)

    #Create dataloaders
    trainloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE
    )

    valloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    return trainloader, valloader


