import torch

from torch import nn


class OptimizedDNN(nn.Module):
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

        super(OptimizedDNN, self).__init__()
        num_emb, dim_emb = embeddings.shape
        
        # 1 - define the embedding layer
        self.embedding = nn.Embedding(num_embeddings = num_emb, embedding_dim = dim_emb) # EX4

        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        self.embedding.weight.data.copy_(torch.from_numpy(embeddings)) # EX4

        # 3 - define if the embedding layer will be frozen or finetuned
        if not trainable_emb:
            self.embedding.weight.requires_grad = False # EX4

        # 4 - define a non-linear transformation of the representations
        self.linear_hid = nn.Linear(2*dim_emb, 256) # EX5
        self.relu = nn.ReLU() # EX5

        # 5 - define the final Linear layer which maps
        # the representations to the classes
        self.linear = nn.Linear(256, output_size) # EX5

    def forward(self, x, lengths, bows):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """

        # 1 - embed the words, using the embedding layer
        embeddings = self.embedding(x)  # EX6

        # 2 - construct a sentence representation out of the word embeddings
        representation_mean = torch.zeros([len(x), embeddings.shape[2]])  # Lab3.1.1
        representation_max = torch.zeros([len(x), embeddings.shape[2]])  # Lab3.1.1
        for i in range(len(x)):
            representation_mean[i] = torch.sum(embeddings[i], dim=0) / lengths[i].float() # Lab3.1.1
            representation_max[i],_ = torch.max(embeddings[i], dim=0)
        representations = torch.cat((representation_mean,representation_max), 1) # Lab3.1.1

        # 3 - transform the representations to new ones.
        representations = self.relu(self.linear_hid(representations.cuda())) # EX6

        # 4 - project the representations to classes using a linear layer
        logits = self.linear(representations) # EX6

        return logits
