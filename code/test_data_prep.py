from collections import OrderedDict
import torch
import pandas as pd
import numpy as np
import pickle
import random
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertConfig, BertModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve, roc_auc_score, precision_score, recall_score, precision_recall_curve
from sklearn.metrics import f1_score


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode(textCol, tokenizer):
  input_ids = []
  attention_mask = []
  for text in textCol:
    tokenized_text = tokenizer.encode_plus(text,
                                          add_special_tokens = True,
                                          max_length = 512,
                                          pad_to_max_length = True,
                                          return_attention_mask = True,
                                          return_tensors = 'pt',
                                          truncation=True
                                           )
    input_ids.append(tokenized_text['input_ids'])
    attention_mask.append(tokenized_text['attention_mask'])

  input_ids = torch.cat(input_ids, dim=0)
  attention_mask = torch.cat(attention_mask, dim=0)

  return input_ids, attention_mask

def get_batches(textCol, labelCol, tokenizer, batch_size=64):
    x = list(textCol.values)

    y_indices = list(labelCol)
    y = torch.tensor(y_indices, dtype=torch.long)
    input_ids, attention_mask = encode(x, tokenizer)
    tensor_dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, y)
    tensor_randomsampler = torch.utils.data.RandomSampler(tensor_dataset)
    tensor_dataloader = torch.utils.data.DataLoader(tensor_dataset, sampler=tensor_randomsampler, batch_size=batch_size)
    return tensor_dataloader

test_df = pd.read_csv("PANC-test.csv", encoding = "latin-1")  #Load PANC dataset 
test_df["label"] = np.where(test_df["label"]=="predator",1,0)

test_dataloader = get_batches(test_df['segment'],test_df['label'], tokenizer, batch_size=8)


torch.cuda.empty_cache()
# If there's a GPU available...
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")



config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
model = BertModel.from_pretrained("bert-base-uncased", config=config)
gpu_model = model.to(device)


with torch.no_grad():
  batches = []
  for i, batch_tuple in enumerate(test_dataloader):
    input_ids, attention_mask, labels = batch_tuple
    outputs = gpu_model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)) #forward does not take labels as an input
    cls = outputs.last_hidden_state[:,0,:].detach().cpu().numpy()

    batches.append((labels, cls))



labels_list = []
embeddings_list = []
for labels, embeddings in batches:
  labels_list.append(labels.numpy())
  embeddings_list.append(embeddings)

#flatten the labels list
all_labels = []
for labels in labels_list:
  for label in labels:
    all_labels.append(label)

#flatten the embeddings list
all_embeddings = []
for embeddings in embeddings_list:
  for embedding in embeddings:
    all_embeddings.append(embedding)



with open('all_embeddings_test.pkl', 'wb') as c: #Add path to file
  pickle.dump(all_embeddings,c)

with open('all_labels_test.pkl', 'wb') as f:  #Add path to file
  pickle.dump(all_labels, f)


