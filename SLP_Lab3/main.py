import os
import warnings

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
import numpy as np
from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN # Prep
from models_1_1 import OptimizedDNN # Lab3.1.1
from models_2_1 import BaselineLSTM # Lab3.2.1
from models_2_2 import OptimizedLSTM # Lab3.2.2
from models_3_1 import AttentionDNN # Lab3.3.1
from models_3_2 import AttentionLSTM # Lab3.3.2
from models_4_1 import OptimizedBiLSTM # Lab3.4.1
from models_4_2 import AttentionBiLSTM # Lab3.4.2
from models_6_1 import BOW_LSTM # Lab3.6.1
from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import sys
import json
from nltk.tokenize import TweetTokenizer

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.50d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 50

EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 50
DATASETS = ["Semeval2017A"]

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# load the raw data
for DATASET in DATASETS:
    if DATASET == "Semeval2017A":
        X_train, y_train, X_test, y_test = load_Semeval2017A()
    elif DATASET == "MR":
        X_train, y_train, X_test, y_test = load_MR()
    else:
        raise ValueError("Invalid dataset")

    # convert data labels from strings to integers
    le = LabelEncoder()

    y_train = le.fit_transform(y_train)  # EX1
    y_test = le.fit_transform(y_test)    # EX1
    n_classes = le.classes_.size         # EX1

    #print("------------------- EX1 -", DATASET, "-------------------")
    #print("The first 10 unencoded labels from the training set are: ")
    #print(le.inverse_transform(y_train[:10]))
    #print("The first 10 encoded labels from the training set are: ")
    #print(y_train[:10],"\n")

    # Define our PyTorch-based Dataset
    bow = sys.argv[2] #Lab3.6.1
    train_set = SentenceDataset(X_train, y_train, word2idx, DATASET, bow)
    test_set = SentenceDataset(X_test, y_test, word2idx, DATASET, bow)
    
    #print("------------------- EX2 -", DATASET, "-------------------")
    #print("The first 10 examples from training set are: ")
    #print(train_set.data[:10],"\n")
    
    #print("------------------- EX3 -", DATASET, "-------------------")
    #for i in range(5):
        #print('dataitem = "', X_train[i], '", label = "', le.inverse_transform(y_train).item(i), '"\n')
        #print("Return values:")
        #print("example = ", train_set[i][0])
        #print("label = ", train_set[i][1])
        #print("length = ", train_set[i][2], "\n")
    
    #la = torch.FloatTensor(train_set[0][3])
    #import pdb; pdb.set_trace()
    # EX7 - Define our PyTorch-based DataLoader
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) # EX7
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True) # EX7

    #############################################################################
    # Model Definition (Model, Loss Function, Optimizer)
    #############################################################################
    model_name = sys.argv[1] # Lab3.1.1 - Lab3.4.2
    model = eval(model_name)(output_size=n_classes, embeddings=embeddings, trainable_emb=EMB_TRAINABLE) # EX8

    # move the mode weight to cpu or gpu
    model.to(DEVICE)
    # save the model - Only for 5.1 question
    '''torch.save(model, model_name+'.pt')'''
    print(model)

    # We optimize ONLY those parameters that are trainable (p.requires_grad==True)
    criterion = torch.nn.BCEWithLogitsLoss() if n_classes == 2 else torch.nn.CrossEntropyLoss() # EX8
    parameters = []  # EX8
    for param in model.parameters():  # EX8
        if param.requires_grad == True: parameters.append(param)  # EX8
    optimizer = torch.optim.Adam(parameters, lr = 0.0001) # EX8

    #############################################################################
    # Training Pipeline
    #############################################################################
    trainning_loss = []
    testing_loss = []
    for epoch in range(1, EPOCHS + 1):
        # train the model for one epoch
        train_dataset(epoch, train_loader, model, criterion, optimizer)

        # evaluate the performance of the model, on both data sets
        train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader, model, criterion)
        '''train_loss, (y_train_gold, y_train_pred), y_train_post, y_train_scores = eval_dataset(train_loader, model, criterion) #lab3.5'''

        test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader, model, criterion)
        '''test_loss, (y_test_gold, y_test_pred), y_test_post, y_test_scores = eval_dataset(test_loader, model, criterion) #lab3.5'''

        # make list of losses to plot them
        trainning_loss.append(train_loss)
        testing_loss.append(test_loss)

    print("----------------- Results for ", DATASET, " dataset -----------------")
    print("The trainning loss is: ", train_loss)
    print("The testing loss is: ", test_loss)
    print("The accuracy, F1score (macro average), recall (macro average) \033[1mfor train set\033[0m are:")
    y_train_gold = np.concatenate(y_train_gold, axis=0)
    y_test_gold = np.concatenate(y_test_gold, axis=0)
    y_train_pred = np.concatenate(y_train_pred, axis=0)
    
    y_test_pred = np.concatenate(y_test_pred, axis=0) 
    '''y_test_post = np.concatenate(y_test_post, axis=0) #lab3.5.2
    y_test_scores = np.concatenate(y_test_scores, axis=0) #lab3.5.2'''
    
    print(classification_report(y_train_gold, y_train_pred), "\n\n")
    print("The accuracy, F1score (macro average), recall (macro average) \033[1mfor test set\033[0m are:")
    print(classification_report(y_test_gold, y_test_pred), "\n")
    
    # create predictions txt file - Only for question 5
    '''with open(model_name+'_predictions.txt', 'w') as f:
        for i in y_test_pred:
            f.write(str(int(i)) + '\n')'''
    
    # create .json file - Only for question 5
    '''tokenizer = TweetTokenizer()
    token_text = [tokenizer.tokenize(example) for example in X_test]
           
    with open(model_name+'.json', 'w') as f:
        all_data = []
        for i in range(len(test_set)):
            jsn_d ={}
            jsn_d['text'] = token_text[i]
            jsn_d['label'] = int(test_set[i][1])
            jsn_d['prediction'] = int(y_test_pred[i])
            jsn_d['posterior'] = y_test_post[i].tolist()
            jsn_d['attention'] = y_test_scores[i].tolist()
            jsn_d['id'] = i
            all_data.append(jsn_d)
        json.dump(all_data, f)'''
    

    #fig = plt.figure()
    plt.plot(trainning_loss, '-o', color = "r", label = "Training data")
    plt.plot(testing_loss, '-o', color = "g", label = "Testing data")
    plt.suptitle('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()