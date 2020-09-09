# Lab1 Overview
## Description of Lab1 steps.
* **Step 1: Create the Corpus**
Just download a corpus from http://www.gutenberg.org __(Gutenberg Project)__.
* **Step 2: Corpus preprocess**
Create your tokenizer and compare it with the tokenizers of the nltk library such as __WhitespaceTokenizer__.
* **Step 3: Create Dictionaries and Alphabets**
Create a dictionary which contains all the different tokens of the corpus and an alphabet which contains all the different characters of the corpus.
* **Step 4: I/O files .syms extension**
Create the appropriate for the FSTs .syms files using the previous alphabet. The format of a .syms file is located here http://www.openfst.org/twiki/pub/FST/FstExamples/ascii.syms
* **Step 5: FST Transducer**
Usage of levenshtein distance. There are three types of edit: 1)__deletion__, 2)__insertion__, 3)__substitution__ each of them have cost 1.

<a href="https://ibb.co/bBKjFNK"><img src="https://i.ibb.co/xj7KYz7/Screenshot-from-2019-11-25-14-59-11.png" alt="Screenshot-from-2019-11-25-14-59-11" border="0"></a>
* **Step 6: FST acceptor**
Create FST acceptor for words and characters like the next image:
<a href="https://ibb.co/F6y7Z7D"><img src="https://i.ibb.co/fMy060x/Screenshot-from-2019-11-25-14-56-52.png" alt="Screenshot-from-2019-11-25-14-56-52" border="0"></a>
* **Step 7: Min Edit Distance Spell Checker**
Acceptor + Transducer = Min Edit Distance Spell Checker
Predictions of the word __cit__: 
1. __sit__
2. __fit__
3. __bit__
4. __it__
5. __city__
6. __with__
7. __win__
* **Step 8: Checker's Evaluation**
Just apply checker on a set of 20 word and check the results.
* **Step 9: Word2Vec representation**
Train 100 2d word2vec embeddings using the Word2Vec class of gensim. The window size is 5 and the #epochs is 1000.
### PART1 - ORTHOGRAPHER
* **Step 10: Export Statistical data**
The word_level and character_level funtions calculate the propability of each word or character respectively.
* **Step 11: FST Transducer again**
The only difference from the step 5 is the weights. There the weight is __weight = - log (Probability)__ where the propability has been calculated in the previous step.
* **Step 12: FST Acceptor again**
Like the step 6 but this time use weights as previously.
* **Step 13: Spell Checker again**
New Spell Checker = New FST Transducer + New FST Acceptor
* **Step 14: Checker's Evaluation again**
The Word-Level spell checher suggests a word which is frequently appered at the text. On the other hand Unigram spell checker suggests a word where the corrected letters are frequently appered at the text.
* **Step 15: Bigram Model**
The procedure is the same with the previous acceptors, transducers anr checker but this time we use the __- log (Probability)__ of each bigram (a pair of two characters).
### PART2 - SENTIMENT ANALYSIS
* **Step 16: Data Preprocess**
Creation of X_train, y_train, X_test, y_test sets. 
* **Step 17: BOW representation**
Representation of the words as one hot encodings and usage of TF-IDF weights (​https://en.wikipedia.org/wiki/Tf–idf​). Consequently we use we use CountVectorizer and TfidfVectorizer of sklearn and using the LogisticRegression we achieve the next results respectively:
* ___Accuracy of training data =  0.9998___
___Accuracy of testing data =  0.8624___


