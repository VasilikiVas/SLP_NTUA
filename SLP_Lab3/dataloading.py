from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from nltk.tokenize import TweetTokenizer
import string
from nltk.corpus import stopwords
from numpy import mean, std
from math import ceil
from sklearn.feature_extraction.text import TfidfVectorizer

#DATASET = "Semeval2017A"  # options: "MR", "Semeval2017A"

class SentenceDataset(Dataset):
    """
    Our custom PyTorch Dataset, for preparing strings of text (sentences)
    What we have to do is to implement the 2 abstract methods:

        - __len__(self): in order to let the DataLoader know the size
            of our dataset and to perform batching, shuffling and so on...

        - __getitem__(self, index): we have to return the properly
            processed data-item from our dataset with a given index
    """
    

    def __init__(self, X, y, word2idx,DATASET, bow):
        """
        In the initialization of the dataset we will have to assign the
        input values to the corresponding class attributes
        and preprocess the text samples

        -Store all meaningful arguments to the constructor here for debugging
         and for usage in other methods
        -Do most of the heavy-lifting like preprocessing the dataset here


        Args:
            X (list): List of training samples
            y (list): List of training labels
            word2idx (dict): a dictionary which maps words to indexes
        """
        # EX2
        self.bow = bow
        if DATASET == "Semeval2017A":
            tweetToken = TweetTokenizer()
            self.data = [tweetToken.tokenize(example) for example in X]

            if bow == "Yes": 
               self.vectorizer = TfidfVectorizer(tokenizer=lambda i:i, lowercase=False)
               self.corpus_tf_idf = self.vectorizer.fit_transform(self.data)
        elif DATASET == "MR":
            self.data = []
            for sample in X:
                new_string = sample.strip()		# remove all the leading and trailing spaces from a string
                new_string = new_string.lower()		# lowercase string
                for punctuation in string.punctuation:	# remove punctuation
                    new_string = new_string.replace(punctuation,' ')
                new_string = "".join((char for char in new_string if char.isalpha() or char.isspace()))		# Keeps only letters and spaces
                new_string = new_string.replace("\n", " ")		# replace newlines with spaces
                new_string = new_string.split()			# use split without parameter to split the words indipendently of spaces number
                self.data.append(new_string)
        else:
            raise ValueError("Invalid dataset")
        self.labels = y
        self.word2idx = word2idx

        # raise NotImplementedError
        
        #EX3
        init_len = [len(sample) for sample in self.data]
        init_len_mean = np.mean(init_len)
        init_len_std = np.std(init_len)
        upper_bound = init_len_mean+2*init_len_std
        lower_bound = init_len_mean-2*init_len_std
        without_outl_len = [l for l in init_len if l >= lower_bound and l <= upper_bound]
        without_outl_len = sorted(without_outl_len)
        self.best_len = without_outl_len[ceil(0.8*len(without_outl_len))]

    def __len__(self):
        """
        Must return the length of the dataset, so the dataloader can know
        how to split it into batches
        Returns:
            (int): the length of the dataset
        """

        return len(self.data)
    

    def __getitem__(self, index):
        """
        Returns the _transformed_ item from the dataset
        Args:
            index (int):
        Returns:
            (tuple):
                * example (ndarray): vector representation of a training example
                * label (int): the class label
                * length (int): the length (tokens) of the sentence
        Examples:
            For an `index` where:
            ::
                self.data[index] = ['this', 'is', 'really', 'simple']
                self.target[index] = "neutral"
            the function will have to return something like:
            ::
                example = [  533  3908  1387   649   0     0     0     0]
                label = 1
                length = 4
        """
        # EX3
        i = 0
        example = np.zeros(self.best_len, dtype = np.int64)
        bowar = []
        for word in self.data[index]:
            if i < self.best_len:
                if self.bow == "Yes":
                   c = self.vectorizer.vocabulary_[word]
                   bowar.append(self.corpus_tf_idf[index].todense()[:,c].item(0))
                if word in self.word2idx.keys():
                   example[i] = (self.word2idx[word])
                else:
                   example[i] = (self.word2idx['<unk>'])
                i += 1
        while(len(bowar)<self.best_len): bowar.append(0)
        bowar = np.asarray(bowar)
        length = len(self.data[index])
        label = self.labels[index]

        return example, label, length, bowar