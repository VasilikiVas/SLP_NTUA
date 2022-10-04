import torch

from torch import nn
import numpy as np


class BOW_LSTM(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    def __init__(self, output_size, embeddings, trainable_emb=False):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        super(BOW_LSTM, self).__init__()
        num_emb, dim_emb = embeddings.shape
        
        # 1 - define the embedding layer
        self.embedding = nn.Embedding(num_embeddings = num_emb, embedding_dim = dim_emb) # EX4

        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        self.embedding.weight.data.copy_(torch.from_numpy(embeddings)) # EX4

        # 3 - define if the embedding layer will be frozen or finetuned
        if not trainable_emb:
            self.embedding.weight.requires_grad = False # EX4

        # 4 - define a lstm transformation of the representations
        #hidden size = 50
        #num of hidden layers = 1
        self.lstm = nn.LSTM(dim_emb, 50, 1, batch_first = True) # Lab3.2.1

        # 5 - define the final Linear layer which maps
        # the representations to the classes
        self.linear = nn.Linear(50, output_size) # EX5

    def forward(self, x, lengths, bows):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """

        # 1 - embed the words, using the embedding layer
        embeddings = self.embedding(x)  # EX6
        
        new_embeddings = torch.zeros([embeddings.shape[0], embeddings.shape[1], embeddings.shape[2]])
        for i in range(embeddings.shape[0]):
            for j in range(embeddings.shape[1]):
                new_embeddings[i][j][:] = bows[i][j]*embeddings[i][j][:]
        
        # 2 - call baseline lstm network
        base_lstm, _ = self.lstm(new_embeddings.cuda()) # Lab3.2.1
        
        # 3 - find the real last timestep 
        real_last_timestep = [min(int(lengths[i]), base_lstm.shape[1])-1 for i in range(len(x))] # Lab3.2.1

        # 4 - construct a sentence representation out of the word embeddings
        representations = torch.zeros([len(x), embeddings.shape[2]])  # EX6     
        for i in range(len(x)):
            representations[i] = base_lstm[i, real_last_timestep[i]]

        # 5 - project the representations to classes using a linear layer
        logits = self.linear(representations.cuda()) # EX6

        return logits
