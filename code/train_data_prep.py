import pandas as pd
import numpy as np
import json
import pickle
import torch
import transformers
from transformers import BertTokenizer, BertConfig, BertModel




train = pd.read_csv("PANC-train.csv", encoding="latin1") #Add path to file
train = train.reset_index()
train["label"] = np.where(train["label"]=="non-predator",0,1)


#Extract name from chatName (drop the number attached to positive conversations)
name_list=[]
for index, row in train.iterrows():
  if (row['label']==1) :
    train.at[index,'name'] = row['chatName'].rsplit('-',1)[0]
  else:
    train.at[index,'name'] = row['chatName']


#In here we will create the dataset splits we are going to use for federated learning. We want to split the data between warm-up/train/test without having the same user in mutliple splits
count_df = train.groupby('name')['label'].count().reset_index().rename(columns={'label':'count'})
label_count_df = pd.merge(count_df, train[['name','label']], how='left', left_on='name', right_on='name')
label_count_df = label_count_df.drop_duplicates()

#Select randomly a number of users by making sure that the total is below a threshold 
#As each predatory user have a different number of data points, we can't just randomly select different users. 
def random_selection(df, totalRows):
  user_list = []
  rows = 0
  for i in df.sample(frac=1).iterrows():
      if (rows + i[1]['count']) <= totalRows:
          rows += i[1]['count']
          user_list.append(i[1]['name'])
      if rows == totalRows:
          break
  return user_list


#Create warm-up data : the warm-up data should represent 10% of the dataset and we want a 50-50 split between positive and negative conversations
num_pos_labels_warmup = 17846 * 10/100/2
num_neg_labels_warmup = 17846 * 10/100/2
pred_warmup_list = random_selection(label_count_df[label_count_df['label']==1], num_pos_labels_warmup)
nonpred_warmup_list = label_count_df[label_count_df['label']==0].sample(n=int(num_neg_labels_warmup))['name'].to_list()
warmup_list = pred_warmup_list + nonpred_warmup_list


#Create validation set
label_count_df_warmup = label_count_df[~(label_count_df['name'].isin(warmup_list))]
#We want 10% of what's left from the training set and that 10% of the datapoints have a predatory label as we want to keep the same distribution as before:
num_pos_labels_val = len(train[~(train['name'].isin(warmup_list))])*0.10*0.10
num_neg_labels_val = len(train[~(train['name'].isin(warmup_list))])*0.10*0.90
pred_val_list = random_selection(label_count_df_warmup[label_count_df_warmup['label']==1], num_pos_labels_val)
nonpred_val_list = label_count_df_warmup[label_count_df_warmup['label']==0].sample(n=int(num_neg_labels_val))['name'].to_list()
val_list = pred_val_list + nonpred_val_list

#Create training set = everything else
train_list = label_count_df_warmup[~label_count_df_warmup['name'].isin(val_list)]['name'].to_list()


#Create the list of non-predatory users in the training set
train_nonpred_list = train[(train['name'].isin(train_list)) & (train['label']==0)]["name"].to_list()


with open('warmup_list.pkl', 'wb') as i: #Add path to file
  pickle.dump(warmup_list, i)
with open('train_list.pkl', 'wb') as l: #Add path to file
  pickle.dump(train_list, l)
with open('val_list.pkl', 'wb') as e: #Add path to file
  pickle.dump(val_list, e)

with open('train_nonpred_list.pkl', 'wb') as f: #Add path to file
  pickle.dump(train_nonpred_list, f)



  #######Feature extraction using BERT
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

  def get_batches(textCol, labelCol, indexCol, tokenizer, batch_size=64):
      x = list(textCol.values)

      y_indices = list(labelCol)
      index_indices = list(indexCol)
      y = torch.tensor(y_indices, dtype=torch.long)
      index = torch.tensor(index_indices, dtype=torch.long)
      input_ids, attention_mask = encode(x, tokenizer)
      tensor_dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, y, index)
      tensor_randomsampler = torch.utils.data.RandomSampler(tensor_dataset)
      tensor_dataloader = torch.utils.data.DataLoader(tensor_dataset, sampler=tensor_randomsampler, batch_size=batch_size)
      return tensor_dataloader
  
  #Create dataloader
  train_dataloader = get_batches(train['segment'],train['label'], train['index'], tokenizer, batch_size=8)

  device = torch.device('cuda')
  torch.cuda.empty_cache()

  config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=False)
  model = BertModel.from_pretrained("bert-base-uncased", config=config)
  gpu_model = model.to(device)


#Get CLS
with torch.no_grad():
  batches = []
  for i, batch_tuple in enumerate(train_dataloader):
    input_ids, attention_mask, labels, index = batch_tuple
    outputs = gpu_model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)) #forward does not take labels as an input
    cls = outputs.last_hidden_state[:,0,:].detach().cpu().numpy()

    batches.append((index, labels, cls))

index_list = []
labels_list = []
embeddings_list = []
for index, labels, embeddings in batches:
  index_list.append(index.numpy())
  labels_list.append(labels.numpy())
  embeddings_list.append(embeddings)


#flatten the labels list
all_labels = []
for labels in labels_list:
  for label in labels:
    all_labels.append(label)

#flatten the index list
all_index = []
for indexes in index_list:
  for index in indexes:
    all_index.append(index)

#flatten the embeddings list
all_embeddings = []
for embeddings in embeddings_list:
  for embedding in embeddings:
    all_embeddings.append(embedding)


#Find the chatnames using the index
df_index = pd.DataFrame(all_index, columns=["index"])
df_merge = pd.merge(df_index, train[['index','chatName']], how="left")
all_names = df_merge.chatName.to_list()

with open('all_embeddings.pkl', 'wb') as i: #Add path to file
  pickle.dump(all_embeddings, i)
with open('all_labels.pkl', 'wb') as l:  #Add path to file
  pickle.dump(all_labels, l)
with open('all_chatnames.pkl', 'wb') as e:  #Add path to file
  pickle.dump(all_names, e)

