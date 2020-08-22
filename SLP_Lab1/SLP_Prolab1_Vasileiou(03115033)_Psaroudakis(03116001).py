#!/usr/bin/env python
# coding: utf-8

# <img src="https://www.rephil.eu/images/emp-in.jpg" width="180" align="center">
# <div align="center">
# <i><br>
# <font size="5"><b>Εθνικό Μετσόβιο Πολυτεχνείο</b></font><br>
# <font size="4">Σχολή Ηλεκτρολόγων Μηχανικών και Μηχανικών Υπολογιστών </font><br>
# <font size="3">Τομέας Σημάτων, Ελέγχου και Ρομποτικής</font><br>
# <font size="3">Εργαστήριο Όρασης Υπολογιστών, Επικοινωνίας Λόγου και Επεξεργασίας Σημάτων</font><br><br>
# <font size="3"><b>Επεξεργασία Φωνής και Φυσικής Γλώσσας</b></font>
# </i><br><br>

# <br><br>
# <div align = "center">
# <i>
# <font size="4"><b>1η Προπαρασκευαστική Άσκηση: Ορθογράφος</b></font><br>
# <font size="4">Ακ. Έτος: 2019 - 2020</font><br>
# <font size="3">Εξάμηνο: 7ο</font><br>
# <br><br><br>
# <font size="3"><b>Βασιλείου Βασιλική - 03115033</font><br>
# <font size="3"><b>Ψαρουδάκης Ανδρέας - 03116001</font>
# </i><br><br><br>

# <font size="4"><b>Βήμα 1: Κατασκευή corpus</b></font>

# Σε αυτό το βήμα θα δημιουργήσουμε ένα μετρίου μεγέθους corpus από διαθέσιμη πηγή στο διαδίκτυο. Συγκεκριμένα, θα κατεβάσουμε το βιβλίο **The Adventures of Sherlock Holmes by Arthur Conan Doyle** από το project Gutenberg, το οποίο αποτελεί μια πηγή για βιβλία που βρίσκονται στο public domain και μία καλή πηγή για γρήγορη συλλογή δεδομένων για κατασκευή γλωσσικών μοντέλων.

# __α)__ Κατεβάζουμε λοιπόν με χρήση της εντολής __wget__ (shell command εντολή) το βιβλίο σε plain txt μορφή και το αποθηκεύουμε με το όνομα __Sherlock.txt__ 

# In[1]:


#Create corpus from "The Adventures of Sherlock Holmes" book
get_ipython().system('wget -c http://www.gutenberg.org/files/1342/1342-0.txt -O Sherlock.txt')


# __β)__ Το corpus που χρησιμοποιούμε έχει έναν ικανοποιητικό αριθμό από διακριτές λέξεις τον οποίο μπορούμε να αυξήσουμε σημαντικά αν προσθέσουμε ακόμη ένα η περισσότερα corpus. Εκτός της άυξησης του όγκου των δεδομένων κάποια άλλα πλεονεκτήμα που προκυπτουν από αυτή την πρακτική είναι:
# 1. Λόγω της ποικιλίας των δεδομένων (λογοτεχνικά κείμενα, ερευνητικά κείμενα, ποιήματα κ.α.) γενικεύουμε ακόμα περισσότερο το μοντέλο μας τόσο γλωσσολογικά όσο και σημασιολογικά. 
# 2. Παράλληλα μειώνουμε το overfit και συνεπώς οι μέθοδοι που χρησιμοποιούμε προσφέρουν καλύτερα και ακριβέστερα στατιστικά στοιχεία για το πρόβλημα που μελετάμε.
# 
# Στην παρούσα εργαστηριακή άσκηση θεωρήσαμε επαρκή την χρήση ενός μόνο corpus καθώς το επιλεχθέν διαθέτει περίπου 7.000 διαφορετικές λέξεις.

# <font size="4"><b>Βήμα 2: Προεπεξεργασία corpus</b></font>

# Αφού πλέον έχουμε κατεβάσει το .txt αρχείο του corpus θα πρέπει να το διαβάσουμε με την κατάλληλη προεπεξεργασία.
# 
# __α)__ Ορίζουμε αρχικά μια συνάρτηση __identity_preprocess__ που διαβάζει ένα string και γυρνάει τον εαυτό του:

# In[2]:


def identity_preprocess(string_):
    return string_


# __β)__ Έπειτα ορίζουμε μια συνάρτηση __read by line__ η οποία δέχεται σαν όρισμα το path του αρχείου μας καθώς και μια συνάρτηση preprocess και διαβάζει το αρχείο γραμμή προς γραμμή σε μία λίστα, καλώντας την preprocess σε κάθε γραμμή. Χρησιμοποιούμε την identity_preprocess του προηγούμενου ερωτήματος σαν default όρισμα για την preprocess:

# In[3]:


#Each line is an item of the list lines
def read_by_line(path, preprocess = identity_preprocess):
    file = open(path, "r")
    lines = []
    for line in file:
        if not line.isspace():
            lines.extend(preprocess(line))
    return lines


# __γ)__ Στη συνέχεια κατασκευάζουμε μια συνάρτηση __tokenize__ η οποία δέχεται σαν όρισμα ένα string s και: 
# 
# α) καλεί την __strip()__ η οποία αφαιρεί όλα τα κενά στην αρχή και στο τέλος του string και την __lower()__ η οποία μετατρέπει όλα τα κεφαλαία γράμματα σε μικρά πάνω στο s, 
# 
# β) αφαιρεί όλα τα σημεία στίξης / σύμβολα / αριθμούς, αφήνωντας μόνο αλφαριθμητικούς χαρακτήρες, 
# 
# γ) αντικαθιστά τα newlines με κένα, 
# 
# δ) κάνει __split()__ τις λέξεις στα κενά. Το αποτέλεσμα είναι μια λίστα από lowercase λέξεις. 

# In[4]:


#Our Tokenizer
def tokenize(s):
    new_s = s.strip()      #remove all the leading and trailing spaces from a string
    new_s = new_s.lower()  #lowercase string
    new_s = "".join((char for char in new_s if char.isalpha() or char.isspace()))   #Keeps only letters and spaces
    new_s = new_s.replace("\n", " ") #replace newlines with spaces
    new_s = new_s.split()  #use split without parameter to split the words indipendently of spaces number
    return new_s


# __δ)__ Τώρα θα πειραματιστούμε τόσο με τον δικό μας tokenizer όσο και με κάποιους __tokenizers της βιβλιοθήκης nltk__. Επιλέγουμε από την βιβλιοθήκη τους __WordPunctTokenizer__ και __WhitespaceTokenizer__ και του κάνουμε import για να τους χρησιμοποιήσουμε. Έπειτα εφαρμόζουμε ενα string και στους 3 tokenizers και συγκρίνουμε τα αποτελέσματα:

# In[5]:


import string
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import WhitespaceTokenizer

s = "   Fai.ry  tales\nof45 yest%er!day, grow but ne&&ver die  "


our_token = tokenize(s)
#Our tokenazier
print("Our tokenizer reuslt is:", our_token,"\n")
#Wordpunct
print("WordPunct tokenizer reuslt is:", WordPunctTokenizer().tokenize(s),"\n")
#Whitespace
print("WhiteSpace tokenizer reuslt is:", WhitespaceTokenizer().tokenize(s),"\n")


# Όπως παρατηρούμε ο tokenizer που εμείς κατασκευάσαμε διαφέρει σε σχέση με τους δύο tokenizers που επιλέξαμε από την βιβλιοθήκη nltk. Πιο συγκεκριμένα, βλέπουμε πως τόσο ο __WordPunct tokenizer__ αλλά και ο __WhiteSpace tokenizer__ δεν αφαιρούν τα σημεία στίξης (τελεία, θαυμαστικό, κόμμα), δεν αφαιρούν τα σύμβολα ( \ , % , & ) ενώ δεν αφαιρούν και τους αριθμούς (45). Επίσης, διαπιστώνουμε ότι ο __WordPunct tokenizer__  κάνει split στα σημεία στίξης, στα σύμβολα και στα newlines. Από την άλλη, ο __WhiteSpace tokenizer__ δεν κάνει split ούτε στα σημεία στίξης αλλά ούτε και στα σύμβολα. Κάνει split στα κενά, στα newlines. Συνεπώς, βλέπουμε πως υπάρχουν αρκετές διαφορές ανάμεσα σε κάθε έναν από τους 3 tokenizers που χρησιμοποιήσαμε. Αυτό είναι λογικό να συμβαίνει καθώς η βιβλιοθήκη nltk περιέχει μια μεγάλη ποικιλία από tokenizers οι οποίοι μπορούν να αξιοποιηθούν για να εξυπηρετήσουν διαφορετικές ανάγκες, οι οποίες εξαρτώνται από την εκάστοτε περίσταση.

# <font size="4"><b>Βήμα 3: Κατασκευή λεξικού και αλφαβήτου</b></font>

# Εδώ θα κατασκευάσουμε __2 λεξικά__ που θα χρησιμεύσουν στην κατασκευή των FSTs, ένα λεξικό για τα tokens (λέξεις) και ένα για τα σύμβολα (αλφάβητο).

# __α)__ Για το __λεξικό για τα tokens__ (λέξεις) ορίζουμε τη συνάρτηση __lexicon__ η οποία δέχεται ως όρισμα το path του αρχείου. 
# Η __lexicon__ χρησιμποιεί την συνάρτηση read_by_line που ορίσαμε στο Βήμα 2 με δεύτερο όρισμα την tokenize. Έτσι, εφαρμόζει τη συνάρτηση tokenize σε κάθε γραμμή του αρχείου επιστρέφοντας εν τέλει μια λίστα με όλες τις λέξεις του αρχείου μας. Έπειτα αφαιρεί τα διπλότυπα, καθώς δεν θέλουμε να έχουμε πανομοιότυπες λέξεις στο λεξικό μας ενώ ταξινομεί τη λίστα μας αλφαριθμητικά. Έτσι λοιπόν, η __lexicon__ επιστρέφει μια ταξινομημένη λίστα με όλες τις λέξεις του αρχείου μας:

# In[6]:


def lexicon(path):
    tokens = read_by_line(path,tokenize)             #fit tokenize function in each line
    dist_tokens = sorted(list(set(tokens)))          #use set function to keep distinct tokens.
                                                     #use list to split into a list of words.
    return dist_tokens


# Έχοντας πλέον ορίσει την __lexicon__ την καλούμε με όρισμα το path του αρχείου ώστε να πάρουμε το λεξικό για τα tokens (__Tokens_Dictionary__):

# In[7]:


Tokens_Dictionary = lexicon("Sherlock.txt")
print(len(Tokens_Dictionary))


# __β)__ Για το __λεξικό για τα σύμβολα__ (αλφάβητο) ορίζουμε τη συνάρτηση __lexicon_syb__ η οποία δέχεται ως όρισμα το path του αρχείου. 
# Η **lexicon_syb** τοποθετεί τα σύμβολα (γράμματα) κάθε μιας λέξης του λεξικού μας σε μια λίστα. Έπειτα αφαιρεί τα διπλότυπα, καθώς δεν θέλουμε να έχουμε πανομοιότυπα σύμβολα στο λεξικό μας ενώ ταξινομεί τη λίστα μας αλφαριθμητικά. Έτσι λοιπόν, η **lexicon_symb** επιστρέφει μια ταξινομημένη λίστα με όλες τα σύμβολα (γράμματα) του αρχείου μας:

# In[8]:


def lexicon_symb(path):
    dist_symb = []
    for token in Tokens_Dictionary:                 #make a list with all letters of all tokens
        dist_symb.extend(token)                     #use set function to get distinct leters
    dist_symb = sorted(list(set(dist_symb)))        #put them in a sorted list
    return dist_symb 


# Έχοντας πλέον ορίσει την __lexicon_syb__ την καλούμε με όρισμα το path του αρχείου ώστε να πάρουμε το λεξικό για τα συμβολα (__Symbol_Dictionary__):

# In[9]:


Symbol_Dictionary = lexicon_symb("Sherlock.txt")


# Το __αλφάβητο__ που προέκυψε από το συκγκεκριμένο κείμενο είναι το ακόλουθο:

# In[10]:


print(Symbol_Dictionary)


# <font size="4"><b>Βήμα 4: Δημιουργία συμβόλων εισόδου / εξόδου</b></font>

# Για την κατασκευή των FSTs χρειάζονται 2 αρχεία που να αντιστοιχίζουν τα σύμβολα εισόδου (ή εξόδου) σε αριθμούς. Ορίζουμε τη συνάρτηση __char_syms_creator__ η οποία δέχεται ως όρισμα τα σύμβολα του αρχείου μας (αλφάβητο). Η __char_syms_creator__ δημιουργεί ένα νέο αρχείο __chars.syms__ στο οποίο αντιστοιχίζει κάθε χαρακτήρα με έναν αύξοντα ακέραιο index. Το πρώτο σύμβολο με index 0 είναι το <epsilon> (ε). Το αρχείο __chars.syms__ έχει τη μορφή: http://www.openfst.org/twiki/pub/FST/FstExamples/ascii.syms 

# In[11]:


def char_syms_creator(symbols):                           #write in file like other ".syms" files
    f = open("chars.syms", 'w')                           #first line includes <epsilon>
    f.write("<epsilon>" + 7*" " + "0" + "\n")             #second line includes <space>
    f.write("<space>" + " " + "1" + "\n")                 #other lines includes the other symbols of Symbol_Dictionary
    num = 2
    for i in range(len(symbols)):
        f.write(symbols[i] + 7*" " + str(num) + "\n")
        num = num +1                                      #increase index
    f.close()


# Έχοντας πλέον ορίσει την __chars_syms_creator__ , την καλούμε με όρισμα το αλφάβητό μας (__Symbol_Dictionary__) ώστε να δημιουργηθεί το αρχείο __chars.syms__ όπως περιγράφηκε παραπάνω:

# In[12]:


Sym_creator = char_syms_creator(Symbol_Dictionary)


# <font size="4"><b>Βήμα 5: Κατασκευή μετατροπέων FST</b></font>

# Για τη δημιουργία του ορθογράφου θα χρησιμοποιήσουμε μετατροπείς βασισμένους στην __απόσταση Levenshtein__. Θα χρησιμοποιήσουμε 3 τύπους από edits: __εισαγωγές χαρακτήρων, διαγραφές χαρακτήρων__ και __αντικαταστάσεις χαρακτήρων__. Κάθε ένα από αυτά τα edits χαρακτηρίζεται από ένα κόστος. Σε αυτό το στάδιο θα θεωρήσουμε ότι όλα τα πιθανά edits έχουν το ίδιο κόστος w=1.
# 
# __α)__ Κατασκευάζουμε λοιπόν ένα μετατροπέα με μία κατάσταση που υλοποιεί την απόσταση Levenshtein αντιστοιχίζοντας: 
# 
# 1. κάθε χαρακτήρα στον εαυτό του με βάρος 0 (no edit), 
# 2. κάθε χαρακτήρα στο epsilon (ε) με βάρος 1 (deletion), 
# 3. το epsilon (ε) σε κάθε χαρακτήρα με βάρος 1 (insertion),
# 4. κάθε χαρακτήρα σε κάθε άλλο χαρακτήρα με βάρος 1 (substitution) 
# 
# Ορίζουμε τη συνάρτηση __fst_transducer__ η οποία δέχεται ως όρισμα τα σύμβολα του αρχείου μας (αλφάβητο). Η __fst_transducer__ δημιουργεί ένα αρχείο __Transducer.txt__ το οποίο περιέχει την περιγραφή του μετατροπέα μας:

# In[13]:


def fst_transducer(symbols):
    f = open("Transducer.txt", 'w')
    for i in range(len(symbols)):
        #no edit
        f.write("0" + " " + "0" + " " + str(symbols[i]) + " " + str(symbols[i]) + " " + "0" + "\n")
        #deletion
        f.write("0" + " " + "0" + " " + str(symbols[i]) + " " + "<epsilon>" + " " + "1" + "\n") 
        #insertion
        f.write("0" + " " + "0" + " " + "<epsilon>" + " " + str(symbols[i]) + " " + "1" + "\n")
        #substitution
        for j in range(len(symbols)):
            if i != j:
                f.write("0" + " " + "0" + " " + str(symbols[i]) + " " + str(symbols[j]) + " " + "1" + "\n")
    #initial state same as the final state
    f.write("0")
    f.close()


# Έχοντας πλέον ορίσει την __fst_transducer__ , την καλούμε με όρισμα το αλφάβητό μας (__Symbol_Dictionary__) ώστε να δημιουργηθεί το αρχείο __Transducer.txt__ όπως περιγράφηκε παραπάνω:

# In[14]:


Traducer = fst_transducer(Symbol_Dictionary)


# Τώρα κάνουμε __compile__ το αρχείο __Transducer.txt__ που δημιουργήσαμε μέσω της εντολής __fstcompile__ (shell command εντολή) παράγοντας ένα binary αρχείο __Transducer.bin.fst__ . 

# In[15]:


get_ipython().system(' fstcompile --isymbols=chars.syms --osymbols=chars.syms Transducer.txt Transducer.bin.fst')


# In[16]:


#! fstdraw --isymbols=chars.syms --osymbols=chars.syms --fontsize=15 -portrait Transducer.bin.fst | dot -Tpdf >Transducer.pdf


# <a href="https://ibb.co/cxc02Zj"><img src="https://i.ibb.co/BwVbTQ9/Screenshot-from-2019-11-25-14-59-11.png" alt="Screenshot-from-2019-11-25-14-59-11" border="0"></a>

# Αν πάρουμε το __shortest path__ για μια λέξη εισόδου του μετατροπέα τότε αυτός θα μας επιστρέψει την __ίδια την λέξη χωρίς καμία αλλαγή (no edits)__. Αυτό προφανώς συμβαίνει γιατί κάθε χαρακτήρας αντιστοιχίζεται στον ευατό του με βάρος 0 (no edit). Οποιαδήποτε άλλη λειτουργία (insertion, deletion, substitution) εχει βάρος (κόστος) 1.

# __β)__ Στην προηγούμενη υλοποίηση θεωρήσαμε ότι για οποιαδήποτε αλλαγή το βάρος είναι ίδιο και ίσο με τη μονάδα Αυτός είναι ένας αρκετά αφελής τρόπος για τον υπολογισμό των βαρών για κάθε edit. Μια πιο σωστή αντιμετώπηση είναι η χρήση βαρών με βάση την συχνότητα εμφάνισης κάθε λάθους. Συγκεκριμένα υπολογίζουμε την πιθανότητα προσθήκης, παράλειψης ή λανθασμένης αντικατάστασης κάθε χαρακτήρα. Στη συνέχεια υπολογίζουμε τα κόστη μέσα από την σχέση __cost = - log(probability)__ . Με αυτό τον τρόπο υπολογίζουμε τα τελικά βάρη για οποιαδήποτε edit.

# <font size="4"><b>Βήμα 6: Κατασκευή αποδοχέα λεξικού</b></font>

# __α)__ Κατασκευάζουμε τώρα έναν αποδοχέα με μία αρχική κατάσταση που αποδέχεται κάθε λέξη του λεξικού από το Βήμα 3α. Τα βάρη όλων των ακμών είναι 0. Αυτό είναι ένας αποδοχέας ο οποίος απλά αποδέχεται μια λέξη αν ανήκει στο λεξικό.
# 
# Ορίζουμε τη συνάρτηση __fst_acceptor__ η οποία δέχεται ως όρισμα τις λέξεις του αρχείου μας (tokens). Η __fst_acceptor__ δημιουργεί ένα αρχείο __Acceptor.txt__ το οποίο περιέχει την περιγραφή του αποδοχέα μας. Συγκεκριμένα σπάει κάθε μία λέξη του λεξικού μας σε γράμματα και δημιουργεί μια νέα κατάσταση για κάθε ένα από αυτά τα γράμματα, ξεκινώντας από την αρχική κατάσταση. Πρακτικά παράγει τόσα μονοπάτια όσες είναι και οι λέξεις του λεξικού μας, ενώ κάθε ένα από αυτά τα μονοπάτια αποτελείται από τόσες καταστάσεις όσα και τα γράμματα της εκάστοτε λέξης.

# In[17]:


def fst_acceptor(tokens):
    f = open("Acceptor.txt", "w")
    state = 1
    for token in tokens:
        letters_ar = list(token)
        for letter in range(0, len(letters_ar)):
            if letter == 0:
                f.write("1" + " " + str(state+1) + " " + str(letters_ar[0]) + " " + str(letters_ar[0]) + " " + "0" + "\n")
            else:
                f.write(str(state) + " " + str(state+1) + " " + str(letters_ar[letter]) + " " + str(letters_ar[letter]) + " " + "0" + "\n")
            state = state + 1
            if letter + 1 == len(letters_ar):
                f.write(str(state) + " " + "0" + " " + "<epsilon>" + " " + "<epsilon>" + " " + "0" + "\n")
    f.write("0")
    f.close()


# Έχοντας πλέον ορίσει την __fst_acceptor__ , την καλούμε με όρισμα το λεξικό μας (__Tokens_Dictionary__) ώστε να δημιουργηθεί το αρχείο __Acceptor.txt__ το οποίο περιέχει την περιγραφή του αποδοχέα μας:

# In[18]:


Acceptor = fst_acceptor(Tokens_Dictionary)


# Τώρα κάνουμε __compile__ το αρχείο __Acceptor.txt__ που δημιουργήσαμε μέσω της εντολής __fstcompile__ (shell command εντολή) παράγοντας ένα binary αρχείο __Acceptor.bin.fst__ . 

# In[19]:


get_ipython().system(' fstcompile --isymbols=chars.syms --osymbols=chars.syms Acceptor.txt Acceptor.bin.fst')


# In[20]:


#! fstdraw --isymbols=chars.syms --osymbols=chars.syms -portrait Acceptor.bin.fst | dot -Tpdf >Acceptor.pdf


# __β)__ Καλούμε τώρα τις συναρτήσεις **fstrmepsilon()**, **fstdeterminize()** και **fstminimize()** για να βελτιστοποιήσουμε το μοντέλο:
# 
# __fstrmepsilon__: Με χρήση της fstrmepsilon() μετατρέπουμε το FST σε ένα ισοδύναμο που __δεν περιέχει ε-μεταβάσεις__.

# In[21]:


get_ipython().system(' fstrmepsilon Acceptor.bin.fst Acceptor.bin.fst')


# __fstdeterminize__: Με χρήση της fstdeterminize() μετατρέπουμε το FST σε ένα ισοδύναμο __ντετερμινιστικό__. Έτσι, από κάθε κατάσταση έχουμε μια μοναδική μετάβαση για κάθε σύμβολο εισόδου.

# In[22]:


get_ipython().system(' fstdeterminize Acceptor.bin.fst Acceptor.bin.fst')


# __fstminimize__: Με χρήση της fstminimize() μετατρέπουμε το FST σε ένα ισοδύναμο που έχει τον __ελάχιστο αριθμό καταστάσεων__.

# In[23]:


get_ipython().system(' fstminimize Acceptor.bin.fst Acceptor.bin.fst')


# In[24]:


#! fstdraw --isymbols=chars.syms --osymbols=cahrs.syms -portrait Acceptor.bin.fst | dot -Tpdf >Minimum_Acceptor.pdf


# Στην επόμενη εικόνα φαίνεται ένας acceptor για 4 ενδεικτικά λέξεις από το κείμενο μας.

# <a href="https://ibb.co/F6y7Z7D"><img src="https://i.ibb.co/fMy060x/Screenshot-from-2019-11-25-14-56-52.png" alt="Screenshot-from-2019-11-25-14-56-52" border="0"></a>

# <font size="1"><b>Σημείωση: Κάνοντας κλικ πάνω στην εικόνα, ανοίγει νέα καρτέλα όπου φαίνεται η εικόνα σε μεγαλύτερες διαστάσεις.</b></font>

# ### Βήμα 7: Κατασκευή ορθογράφου

# Συνθέτουμε τώρα τον Levenshtein transducer με τον αποδοχέα του ερωτήματος 6α παράγοντας τον min edit distance spell checker. Αυτός ο transducer διορθώνει τις λέξεις χωρίς να λαμβάνει υπόψιν του κάποια γλωσσική πληροφορία, με κριτήριο να κάνει τις ελάχιστες δυνατές μετατροπές στην λέξη εισόδου. Θα αναλύσουμε την συμπεριφορά του μετατροπέα στην περίπτωση που:
# 
# **α)** τα edits είναι ισοβαρή

# Πρώτα ταξινομούμε τις εξόδους του transducer με χρήση της συνάρτησης __fstarcsort__ παράγοντας το αρχείο __Τransducer_sorted.fst__

# In[25]:


get_ipython().system(' fstarcsort --sort_type=olabel Transducer.bin.fst Τransducer_sorted.fst')


# Στη συνέχεια ταξινομούμε τις εισόδους του acceptor με χρήση της συνάρτησης __fstarcsort__ παράγοντας το αρχείο __Αcceptor_sorted.fst__ .

# In[26]:


get_ipython().system(' fstarcsort --sort_type=ilabel Acceptor.bin.fst Αcceptor_sorted.fst')


# Έπειτα, χρησιμοποιούμε τη συνάρτηση **fstcompose()** για τη σύνθεση του transducer με τον acceptor αποθηκεύοντας τον min edit distance spell checker στο αρχείο __spell_checker.fst__.

# In[27]:


get_ipython().system(' fstcompose Τransducer_sorted.fst Αcceptor_sorted.fst spell_checker.fst')


# Η συμπεριφορά του μετατροπέα που έχουμε υλοποιήσει διαφέρει ανάλογα με τα βάρη που θα βάλουμε σε κάθε edit. Συγκεκριμένα:
# 1. Στην περίπτωση που τα edits είναι ισοβαρή ο transducer λειτουργεί με μοναδικό κριτήριο τις ελάχιστες μεταρτοπές στην λέξη. Δηλαδή αρχικά αναζητά λέξεις που διαφέρουν μόνο κατά ένα γράμμα και αν δεν βρει τέτοια οδηγείται σε αυτές με δύο και ούτω καθεξής.
# 2. Στην περίπτωση τώρα που τα edits δεν είναι ισοβαρή προκύπτει ένας πιο σωστός trandsucer. Συγκεκριμένα γίνεται χρήση βαρών με βάση την συχνότητα εμφάνισης κάθε λάθους. Υπολογίζουμε την πιθανότητα προσθήκης, παράλειψης ή λανθασμένης αντικατάστασης κάθε χαρακτήρα. Στη συνέχεια υπολογίζουμε τα κόστη μέσα από την σχέση __cost = - log(probability)__ . Με αυτό τον τρόπο υπολογίζουμε τα τελικά βάρη για οποιαδήποτε edit και έτσι επιτυγχάνεται μια διόρθωση πιο συναφής με αυτό που ο χρήστης θέλει να γράψει και συνεπώς έχουμε μεγαλύτερα ποσοστά επιτυχίας.

# __β)__ Θέλουμε τώρα να βρούμε τις πιθανές προβλέψεις του min edit spell checker αν η είσοδος είναι η λέξη __cit__.
# 
# Ορίζουμε τη συνάρτηση __word_fst__ η οποία δέχεται σαν όρισμα μια λέξη (word). H __word_fst__ δημιουργεί ένα αρχείο __Word.txt__ στο οποίο και γράφει την περιγραφή ενός μετατροπέα FST που αποδέχεται την λέξη "cit". 

# In[28]:


def word_fst(word):
    f = open("Word.txt", 'w')
    state = 0
    word_ar = list(word)
    for letter in word_ar:
        f.write(str(state) + " " + str(state + 1) + " " + letter + " "  + "\n")
        state = state + 1
    f.write(str(state) + "\n")
    f.close()

word_fst("cit")


# In[29]:


get_ipython().system(' fstcompile --isymbols=chars.syms --acceptor=true Word.txt Word.bin.fst')
get_ipython().system(' fstdraw --isymbols=chars.syms --osymbols=chars.syms -portrait Word.bin.fst | dot -Tjpg >Cit.jpg')
get_ipython().system(' fstcompose Word.bin.fst spell_checker.fst corrections.fst')
get_ipython().system(' fstshortestpath corrections.fst corrections.fst')
get_ipython().system(' fstrmepsilon corrections.fst corrections.fst')
get_ipython().system(' fstprint --isymbols=chars.syms --osymbols=chars.syms corrections.fst')
get_ipython().system(' fstdraw --isymbols=chars.syms --osymbols=chars.syms -portrait corrections.fst | dot -Tjpg >Correct.jpg')
print('\n')
print("Instead of 'cit', do you mean:")
get_ipython().system(' ./pred.sh spell_checker.fst')


# Χρησιμοποιώντας το shortest path μας δίνεται κάθε φορά μια πιθανή λέξη διόρθωσης. Στη συνέχεια αφαιρώντας αυτή τη λέξη από το λεξικό και ακολουθώντας την ίδια διαδικασία προκύπτουν άλλες πιθανές λέξεις. Εμείς εκτελέσαμε αυτή την διαδικασία άλλες επτά φορές και προέκυψαν με την σειρά οι εξής λέξεις:
# 1. __sit__
# 2. __fit__
# 3. __bit__
# 4. __it__
# 5. __city__
# 6. __with__
# 7. __win__
# 
# Παρατηρούμε ότι οι πέντε πρώτες λέξεις καθώς και η wit που προέκυψε στην πρώτη φορά εκτέλεσης διαφέρουν μόνο κατά ενα γράμμα (αντικατάσταση, προσθήκη, αφαίρεση). Ενώ οι λέξεις έξι και επτά διαφέρουν κατά δύο γράμματα. Επιβεβαιώνουμε λοιπόν με αυτό τον τρόπο την λειτουργία του transduser με ισοβαρή edits όπως περιγράψαμε στο προηγούμενο ερώτημα.
# 
# Υπάρχουν ακόμα πολλές λέξεις που μοιάζουν με τη λέξη cit όπως είναι οι λέξεις cut, cat, cite όμως όπως έχουμε ήδη προαναφέρει χρησιμοποιήθηκε ένα μετρίου μεγέθους corpus οπότε είναι πιθανό πολλές λέξεις να μην βρίσκονται στο λεξιλόγιο μας και συνεπώς να μην μπορούν να προβλεφθούν.

# <font size="4"><b>Βήμα 8: Αξιολόγηση ορθογράφου</b></font>

# __α)__ Κατεβάζουμε τώρα με χρήση της εντολής __wget__ (shell command εντολή) το σύνολο δεδομένων για το evaluation και το αποθηκεύουμε στο αρχείο __Tester.txt__ 

# In[30]:


get_ipython().system('wget -c https://raw.githubusercontent.com/georgepar/python-lab/master/spell_checker_test_set -O Tester.txt')


# __β)__ Ανοίγουμε το αρχείο __Tester.txt__ και αποθηκεύουμε σε μια λίστα (words) όλες τις δυνατές προς διόρθωση λέξεις. Στη συνέχεια, __επιλέγουμε τυχαία 20__ , με σκοπό να κάνουμε σε αυτές την __βέλτιστη διόρθωση__, όπως αυτή προβλέπεται με βάση τον αλγόριθμο ελαχίστων μονοπατιών στο γράφο του μετατροπέα του Βήματος 7.

# In[31]:


import random
file = open("Tester.txt", "r")
words = []
random_words = []
for line in file:
    if not line.isspace():
        line = line.replace("\n", " ")
        line = line.split(":")[1]
        word = line.split()
        words = word + words
    
print("Whe choose randomly the following 20 words: \n")
for i in range(20):
    random_words.append(random.choice(words))
    print(random_words[i])


# Οι παραπάνω 20 λέξεις, όπως είπαμε, επιλέχθηκαν τυχαία από το σύνολο των δεδομένων που είχαμε στη διάθεση μας. Τώρα για κάθε μία από αυτές τις λέξεις καλούμε την συνάρτηση __word fst__ που ορίσαμε στο Βήμα 7β ώστε να τις διορθώσουμε πραγματοποιώντας τον ελάχιστο αριθμό αλλαγών (edits). Το αρχείο __pred.sh__ το οποίο χρησιμοποιούμε χρησιμεύει για την τύπωση των διορθωμένων λέξεων:

# In[32]:


for word in random_words:
    print(word + ":" + " ",end='')
    word_fst(word)
    get_ipython().system(' ./pred.sh spell_checker.fst')
    print("\n")


# __Παρατηρήσεις__: 
# 
# Παρατηρούμε πως σε πολλές περιπτώσεις ο ορθογράφος μας κάνει διορθώσεις που αλλοιώνουν το νόημα των λέξεων , με αποτέλεσμα να μην δίνει τα αναμενόμενα αποτελέσματα. Για παράδειγμα, η λέξη __bycycle__ διορθώνεται σε __style__ και όχι σε __bicycle__ οπως θα περίμενε κανείς, εφόσον με αλλαγή ενός μόνο χαρακτήρα (κόστος 1) θα προέκυπτε η λέξη __bicycle__ . Αντίστοιχα, η λέξη __intial__ διορθώνεται στη λέξη __until__ , κάτι το οποίο προϋποθέτει 3 edits ενώ μια προτιμότερη διόρθωση θα ήταν η λέξη __initial__ , η οποία και θα μπορούσε να προκύψει με 1 μόνο εισαγωγή του χαρακτήρα που λείπει (1 edit). Αυτά τα "σφάλματα" οφείλονται στο γεγονός ότι το corpus μας είναι μικρό και δεν περιέχει αυτές τις συγκεκριμένες λέξεις που θα αποτελούσαν μια καλύτερη διόρθωση. Αν είχαμε ένα μεγαλύτερο όγκο δεδομένων στη διάθεση μας τότε τα αποτελέσματα θα ήταν σαφώς καλύτερα, αν και μπορούμε να πούμε ότι και τώρα σε κάποιες περιπτώσεις η διόρθωση λειτουργεί ικανοποιητικά.

# <font size="4"><b>Βήμα 9: Εξαγωγή αναπαραστάσεων word2vec</b></font>

# __α)__ Διαβάζουμε το κείμενο μας σε μια λίστα από tokenized προτάσεις αξιοποιώντας την συνάρτηση __tokenize__ του Βήματος 2γ:

# In[33]:


import re
file = open("Sherlock.txt", "r")
text = file.read()
token_text = []
sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
for stuff in sentences:
    token_set = tokenize(stuff)
    token_text.append(token_set)


# __β)__ Χρησιμοποιούμε τώρα την κλάση __Word2Vec__ του __gensim__ για να εκπαιδεύσουμε 100 διάστατα __word2vec embeddings__ με βάση τις προτάσεις του προηγούμενου βήματος. Χρησιμοποιούμε __window=5__ και __1000 εποχές__:

# In[34]:


from gensim.models import Word2Vec

# Initialize word2vec. Context is taken as the 2 previous and 2 next words
model = Word2Vec(token_text, window=5, size=100, workers=4)
model.train(token_text, total_examples=len(sentences), epochs=1000)

# get ordered vocabulary list
voc = model.wv.index2word

# get vector size
dim = model.vector_size

# Convert to numpy 2d array (n_vocab x vector_size)
def to_embeddings_Matrix(model):  
    embedding_matrix = np.zeros((len(model.wv.vocab), model.vector_size))
    word2idx = {}
    for i in range(len(model.wv.vocab)):
        embedding_matrix[i] = model.wv[model.wv.index2word[i]]
        word2idx[model.wv.index2word[i]] = i
    return embedding_matrix, model.wv.index2word, word2idx


# __γ)__ Επιλέγουμε τώρα __10 τυχαίες λέξεις__ από το λεξικό και βρίσκουμε τις __σημασιολογικά κοντινότερες__ τους.

# In[35]:


import random

random_words = random.sample(voc, 10)
for word in random_words:
    # get most similar words
    sim = model.wv.most_similar(word, topn=5)
    print('"' + word + '"' + " is similar with the words:" , '\n')
    for s in sim:
        print('"' + s[0] + '"' + " with similarity " + str(s[1]))
    print('\n')


# Θα δοκιμάσουμε να **αυξήσουμε το μέγεθους του παραθύρου μας (__window__) από 5 σε 10** διατηρώντας σταθερές τις υπόλοιπες παραμέτρους του μοντέλους μας:

# In[36]:


# Initialize word2vec. Context is taken as the 2 previous and 2 next words
model = Word2Vec(token_text, window=10, size=100, workers=4)
model.train(token_text, total_examples=len(sentences), epochs=1000)

# get ordered vocabulary list
voc = model.wv.index2word

# get vector size
dim = model.vector_size

# Convert to numpy 2d array (n_vocab x vector_size)
def to_embeddings_Matrix(model):  
    embedding_matrix = np.zeros((len(model.wv.vocab), model.vector_size))
    word2idx = {}
    for i in range(len(model.wv.vocab)):
        embedding_matrix[i] = model.wv[model.wv.index2word[i]]
        word2idx[model.wv.index2word[i]] = i
    return embedding_matrix, model.wv.index2word, word2idx


# In[37]:


random_words = random.sample(voc, 10)
for word in random_words:
    # get most similar words
    sim = model.wv.most_similar(word, topn=5)
    print('"' + word + '"' + " is similar with the words:" , '\n')
    for s in sim:
        print('"' + s[0] + '"' + " with similarity " + str(s[1]))
    print('\n')


# Τέλος θα επιχειρήσουμε να __αυξήσουμε τον αριθμό των εποχών από 1000 σε 2000__ διατηρώντας τις άλλες παραμέτρους του μοντέλου μας σταθερές:

# In[40]:


# Initialize word2vec. Context is taken as the 2 previous and 2 next words
model = Word2Vec(token_text, window=5, size=100, workers=4)
model.train(token_text, total_examples=len(sentences), epochs=2000)

# get ordered vocabulary list
voc = model.wv.index2word

# get vector size
dim = model.vector_size

# Convert to numpy 2d array (n_vocab x vector_size)
def to_embeddings_Matrix(model):  
    embedding_matrix = np.zeros((len(model.wv.vocab), model.vector_size))
    word2idx = {}
    for i in range(len(model.wv.vocab)):
        embedding_matrix[i] = model.wv[model.wv.index2word[i]]
        word2idx[model.wv.index2word[i]] = i
    return embedding_matrix, model.wv.index2word, word2idx


# In[41]:


random_words = random.sample(voc, 10)
for word in random_words:
    # get most similar words
    sim = model.wv.most_similar(word, topn=5)
    print('"' + word + '"' + " is similar with the words:" , '\n')
    for s in sim:
        print('"' + s[0] + '"' + " with similarity " + str(s[1]))
    print('\n')


# __Παρατηρήσεις__: 
# 
# Παρατηρούμε ότι τα αποτελέσματα δεν είναι τόσο ποιοτικά όσο περιμέναμε. Αυξάνοντας το μέγεθος του παραθύρου (window) ή τον αριθμό των εποχών (epochs) βλέπουμε ότι αυτά βελτιώνονται λίγο μιας και εκπαιδεύουμε καλύτερα το μοντέλος μας, ωστόσο πάλι δεν είναι ιδιαίτερα ποιοτικά. Αξίζει να σημειωθεί ότι δεν μπορούμε να αυξήσουμε πάρα πολύ το μέγεθος του παραθύρου μας καθώς σε αυτή την περίπτωση ενδέχεται να βγούμε έξω από τα συντακτικά/σημασιολογικά όρια μιας λέξης και να έχουμε ανεπιθύμητα αποτελέσματα. Αντίθετα, μπορούμε να αυξήσουμε χωρίς κάποιο περιορισμό των αριθμό των εποχών ώστε να λάβουμε καλύτερα αποτελέσματα. Αυτό ωστό θα αυξήσει σημαντικά το χρόνο εκπαίδευσης του μοντέλου μας. Για να μπορούσαμε να λάβουμε αρκετά πιο ποιοτικά αποτελέσματα θα έπρεπε να έχουμε και ένα μεγαλύτερο corpus στην διάθεση μας.

# In[ ]:




