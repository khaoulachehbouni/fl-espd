# Early Detection of Sexual Predators with Federated Learning


In this project, we train a model in a federated setting to detect online grooming in conversations. You can refer to the following workshop paper for more information about our implementation: https://openreview.net/pdf?id=M84OnT0ZvDq

<h2> Dataset </h2>
Our work is an extension of https://early-sexual-predator-detection.gitlab.io/, please refer to their publicly available code for the creation of the PANC dataset, the dataset used in our experiments. The PANC dataset requires access to the PAN12 dataset, that can be obtained from Zenodo (https://zenodo.org/record/3713280#.Y4_XdXbMLMY), and access to the ChatCoder 2 dataset which is available by request (https://www.chatcoder.com/data.html).


<h2> Data Manipulation</h2>
Before being able to train our model, some necessary steps are required.
First, the create_panc_tvt.py file allows us to create the augmented data chunks required for training. 
Once this step is validated, the data_split.py file allows us to split the training dataset into a training set, a warm-up set and a validation set.

<h2>Training</h2>
In our experiments, we decided to train a logistic regression model in a federated manner but using BERT embeddings. However, for computational reason, we perform the features extraction in a centralized manner: the file bert_feature_extraction.py allows us to extract the necessary embeddings from the training set.

The files ClientDP.py and utilsDP.py contains our federated learning implementation with differential privacy

<h2>Evaluation</h2>
The files annotate_datapack_with_predictions_FL.py, eval_util.py and message_based_evaluation.py were files present in Vogt et al.'s implementation (https://early-sexual-predator-detection.gitlab.io/) that we have adapted to suit our federated set-up. 
