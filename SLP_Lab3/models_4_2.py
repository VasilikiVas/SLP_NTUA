import torch
from SelfAttention import SelfAttention # Lab3.3.1
from torch import nn


class AttentionBiLSTM(nn.Module):

    def __init__(self, output_size, embeddings, trainable_emb=False):
        super(AttentionBiLSTM, self).__init__()
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
        self.Bi_lstm = nn.LSTM(dim_emb, 50, 1, batch_first = True, bidirectional = True) # Lab3.4.2

        # 5 - define the final Linear layer which maps
        # the representations to the classes
        self.linear = nn.Linear(2*50, output_size) # EX5
        self.attention = SelfAttention(2*50, batch_first = True, non_linearity = "tanh") # Lab3.4.2

    def forward(self, x, lengths, bows):

        # 1 - embed the words, using the embedding layer
        embeddings = self.embedding(x)  # EX6

        # 2 - call baseline lstm network
        base_Bilstm, _ = self.Bi_lstm(embeddings) # Lab3.4.2

        # 3 - call attention to get the representations.
        representations = self.attention(base_Bilstm, lengths) # Lab3.4.2
        '''representations, scores = self.attention(base_Bilstm, lengths) # Lab3.5'''
        
        # 4 - project the representations to classes using a linear layer
        logits = self.linear(representations) # EX6

        return logits, scores
        '''return logits, scores # Lab3.5'''
