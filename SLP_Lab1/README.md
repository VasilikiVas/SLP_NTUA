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
* **Step 9: Word2Vec representation
Train 100 2d word2vec embeddings using the Word2Vec class of gensim. The window size is 5 and the #epochs is 1000.
