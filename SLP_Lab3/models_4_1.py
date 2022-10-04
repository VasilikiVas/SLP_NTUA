import torch

from torch import nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class OptimizedBiLSTM(nn.Module):
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

        super(OptimizedBiLSTM, self).__init__()
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
        self.Bi_lstm = nn.LSTM(dim_emb, 50, 1, batch_first = True, bidirectional = True) # Lab3.2.1

        # 5 - define the final Linear layer which maps
        # the representations to the classes
        self.linear = nn.Linear(4*50, output_size) # Lab3.2.2

    def forward(self, x, lengths, bows):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """

        # 1 - embed the words, using the embedding layer
        embeddings = self.embedding(x)  # EX6
        
        # 2 - call baseline lstm network
        base_Bilstm, _ = self.Bi_lstm(embeddings) # Lab3.2.2
        
        # 3 - find the real last timestep 
        real_last_timestep = [min(int(lengths[i]), base_Bilstm.shape[1])-1 for i in range(len(x))] # Lab3.2.2
        
        # 4 - Limit of forward and backward
        limit = int(base_Bilstm.shape[2]/2)

        # 5 - construct a sentence representation out of the word embeddings
        representations_Bilstm_fw = torch.zeros([len(x), embeddings.shape[2]])  # Lab3.4.1     
        representations_Bilstm_bw = torch.zeros([len(x), embeddings.shape[2]])  # Lab3.4.1

        for i in range(len(x)):
            representations_Bilstm_fw[i] = base_Bilstm[i, real_last_timestep[i], :limit]  # Lab3.4.1
            representations_Bilstm_bw[i] = base_Bilstm[i, real_last_timestep[i], limit:]  # Lab3.4.1
        representation_mean = torch.zeros([len(x), embeddings.shape[2]])  # Lab3.2.2
        representation_max = torch.zeros([len(x), embeddings.shape[2]])  # Lab3.2.2
        for i in range(len(x)):
            representation_mean[i] = torch.sum(embeddings[i], dim=0) / lengths[i].float() # Lab3.2.2
            representation_max[i],_ = torch.max(embeddings[i], dim=0)
        
        representations = torch.cat((representations_Bilstm_fw, representations_Bilstm_bw, representation_mean,representation_max), 1) # Lab3.4.1
        # 6 - project the representations to classes using a linear layer
        logits = self.linear(representations.cuda()) # EX6

        return logits
