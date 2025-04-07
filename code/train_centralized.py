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
from sklearn.linear_model import LogisticRegressionCV
import argparse


parser = argparse.ArgumentParser(description="Centralized Training LR")

#Simulation parameters
parser.add_argument("--split", default="all") #warmup or all



if __name__ == "__main__":

    args = parser.parse_args()

    #Load the data
    with open('all_embeddings.pkl', 'rb') as w: #Add the path to the file
        all_embeddings_bert = pickle.load(w)

    with open('all_chatnames.pkl', 'rb') as t:#Add the path to the file
        all_names_bert = pickle.load(t)

    with open('all_labels.pkl', 'rb') as v: #Add the path to the file
        all_labels_bert = pickle.load(v)

    #Create a dataframe
    df = pd.DataFrame(list(zip(all_names_bert, all_embeddings_bert, all_labels_bert)), columns=['names','embeddings','labels'])


    if args.split == 'warmup':
        with open('warmup_list.pkl', 'rb') as l:  #Add the path to the file
            warmup_list = pickle.load(l)
            
        train_features = df[df["names"].isin(warmup_list)].embeddings.to_list()
        train_labels = df[df["names"].isin(warmup_list)].labels.to_list()
    else:
        train_features = df.embeddings.to_list()
        train_labels = df.labels.to_list()



    model = LogisticRegressionCV(cv=5, random_state=0)
    model.fit(train_features, train_labels)




    #Evaluate

    #Load the embeddings+labels -- they were created using feature_extraction.py 
    with open('all_embeddings_test.pkl', 'rb') as c:  
        all_embeddings = pickle.load(c)

    with open('all_labels_test.pkl', 'rb') as f:
        all_labels = pickle.load(f)




    #Load the model for evaluation
    # model = pickle.load(open("lr_centralized_bert.sav", 'rb'))

    # test
    y_pred = model.predict(all_embeddings)



    # performance
    print(f'Accuracy Score: {accuracy_score(all_labels,y_pred)}')
    print(f'Confusion Matrix: \n{confusion_matrix(all_labels, y_pred)}')

    print(f'F1 score: {f1_score(all_labels,y_pred)}')
    print(f'Recall score: {recall_score(all_labels,y_pred)}')
    print(f'Precision: {precision_score(all_labels,y_pred)}')
    print(f'Area Under Curve: {roc_auc_score(all_labels, y_pred)}')


    model_file = "lr_warmup_bert.sav" #Add path to file
    pickle.dump(model, open(model_file, 'wb'))