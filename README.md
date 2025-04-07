# Enhancing Privacy in the Early Detection of Sexual Predators Through Federated Learning and Differential Privacy


In this project, we train a model in a federated setting to detect online grooming in conversations. 
You can refer to the following AAAI paper for more information about our implementation: https://arxiv.org/abs/2501.12537
The folder code contains the code described below. The folder workshop contains the code we used for our 2022 NeurIPS workshop paper Early detection of sexual predators with federated learning.

<h2> Dataset </h2>
Our work is an extension of https://early-sexual-predator-detection.gitlab.io/, please refer to their publicly available code for the creation of the PANC dataset, the dataset used in our experiments. The PANC dataset requires access to the PAN12 dataset, that can be obtained from Zenodo (https://zenodo.org/record/3713280#.Y4_XdXbMLMY), and access to the ChatCoder 2 dataset which is available by request (https://www.chatcoder.com/data.html).


<h2> Data Preparation</h2>
Before being able to train our model, some necessary steps are required.

Run the train_data_prep.py and the test_data_prep.py files to get the appropriate data split for training in a federated learning setting: Create the warm-up data, train and validation data and use BERT Base to get an embedding representation of the training set and test set. As fine-tuning BERT directly is too costly in a federated learning scenario, we extract the embeddings centrally and then use a logistic regression for the federated training. 

<h2>Training</h2>
Before any privacy implementations, we train models to ensure that our federated learning setting is not hurting the utility.
As a baseline, we train a centralized model, a model trained on the warm-up data only and a federated model.
The file train_centralized.py contains the code needed to train a model centrally on the full training set or on the training data only (you need to use the data split as an argument).
The files clientCombined.py and utilsSim.py contains the code needed to train a model using federated learning with the Flower library. 
You can run the code using:

```
python clientCombined.py --model_version name_of_model --output_dir path_to_store_model

```
You can also change the other hyperparameter (number of clients, rounds of training, fraction of the clients that participate in the training. The default hyperparameters are the one used in the paper. 

<h2>Training with Privacy</h2>
The default hyperparameters in the code are the one used for the results presented in the paper.

<h3>Training with DP-SGD</h3>
The files clientDP.py and utilsDP.py contains our federated implementation with DPSGD 
The simulation parameters are the same as for the federated learning implementation. The privacy parameters are the learning rate, the number of epochs, and the max gradient norm, epsilon and delta.  

<h3>Training with DP-FedAvg</h3>
The files clientDPFedAvg.py and utilsSim.py contains our federated implementation with DP-FedAvg.
The simulation parameters are the same as for the federated learning implementation. The privacy parameters are the clip norm, noise multiplier and server side noise.

<h3>Training with Metric-DP</h3>
The files clientLDP.py and utilsLDP.py contains our federated implementation with DP-FedAvg.
The simulation parameters are the same as for the federated learning implementation. There is no privacy parameters are we create the noisy embeddings in a centralized way using the file ldp_noise.py. The argument noisy embeddings allows us to load the appropriate file. 


<h2>Evaluation</h2>
The files annotate_datapack_with_predictions_FL_sav.py, annotate_datapack_with_predictions_FL_torch.py, eval_util.py and message_based_evaluation.py were files present in Vogt et al.'s implementation (https://early-sexual-predator-detection.gitlab.io/) that we have adapted to suit our federated set-up. Use annotate_datapack_with_predictions_FL_sav.py with sav models and annotate_datapack_with_predictions_FL_torch.py with torch models. 

