'''
Created on Oct 17, 2016

@author: PRATIK SHAH
'''
# coding: utf-8


"""
CS579: Assignment 2

In this assignment, you will build a text classifier to determine whether a
movie review is expressing positive or negative sentiment. The data come from
the website IMDB.com.

You'll write code to preprocess the data in different ways (creating different
features), then compare the cross-validation accuracy of each approach. Then,
you'll compute accuracy on a test set and do some analysis of the errors.

The main method takes about 40 seconds for me to run on my laptop. Places to
check for inefficiency include the vectorize function and the
eval_all_combinations function.

Complete the 14 methods below, indicated by TODO.

As usual, completing one method at a time, and debugging with doctests, should
help.
"""

# No imports allowed besides these.
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request
import string
import zipfile
import pandas as pd
import io

def download_data():
    """ Download and unzip data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/xk4glpk61q3qrg2/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()


def read_data(path):
    """
    Walks all subdirectories of this path and reads all
    the text files and labels.
    DONE ALREADY.

    Params:
      path....path to files
    Returns:
      docs.....list of strings, one per document
      labels...list of ints, 1=positive, 0=negative label.
               Inferred from file path (i.e., if it contains
               'pos', it is 1, else 0)
    """
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    #for f in sorted(fnames):
      #  print (f)
     #   data=(1, open(f).readlines()[0])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])

def read_data_test(path):
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'test', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
  #  data = [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    return np.array([d[1] for d in data])

def tokenize(doc, keep_internal_punct=False):
    """
    Tokenize a string.
    The string should be converted to lowercase.
    If keep_internal_punct is False, then return only the alphanumerics (letters, numbers and underscore).
    If keep_internal_punct is True, then also retain punctuation that
    is inside of a word. E.g., in the example below, the token "isn't"
    is maintained when keep_internal_punct=True; otherwise, it is
    split into "isn" and "t" tokens.

    Params:
      doc....a string.
      keep_internal_punct...see above
    Returns:
      a numpy array containing the resulting tokens.

    >>> tokenize(" Hi there! Isn't this fun?", keep_internal_punct=False)
    array(['hi', 'there', 'isn', 't', 'this', 'fun'], 
          dtype='<U5')
    >>> tokenize("Hi there! Isn't this fun? ", keep_internal_punct=True)
    array(['hi', 'there', "isn't", 'this', 'fun'], 
          dtype='<U5')
    >>> tokenize("??necronomicon?? geträumte sünden.<br>Hi", True)
    array(['necronomicon', 'geträumte', 'sünden.<br>hi'], 
          dtype='<U13')
    >>> tokenize("??necronomicon?? geträumte sünden.<br>Hi", False)
    array(['necronomicon', 'geträumte', 'sünden', 'br', 'hi'], 
          dtype='<U12')
    """
    token_list=[]
    removeextra=string.punctuation
    if(keep_internal_punct==True):
        split_text = doc.lower().split()
    
       
        for textval in split_text:
            textval=textval.lstrip(removeextra)
            textval=textval.rstrip(removeextra)
            token_list.append(textval)
                    #array = np.array(re.findall(r"[\w']+|[.,!?;^\w]&[^\w\s]", final_text))
        array=np.array(token_list)
    if(keep_internal_punct==False):
        split_text = doc.lower().split()
        final_text =' '.join(split_text)
        array = np.array(re.findall(r"[\w]+|[^\w\s]&[^\w\s]", final_text))
  
    return array
    ###TODO
    pass


def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    Note that the feats dict is modified in place,
    so there is no return value.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_features(['hi', 'there', 'hi'], feats)
    >>> sorted(feats.items())
    [('token=hi', 2), ('token=there', 1)]
    """
    counter=1
    #tokens.sort();
    for i in range(len(tokens)):
        str="token="+tokens[i]
        if( str in feats):
            counter=feats['token='+tokens[i]]
            feats['token='+tokens[i]]=counter+1
            
        else:
            feats['token='+tokens[i]]=1
            #counter=1
    ###TODO
    pass


def token_pair_features(tokens, feats, k=3):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.

    For example [a, b, c, d] with k=3 will consider the
    windows: [a,b,c], [b,c,d]. In the first window,
    a_b, a_c, and b_c appear; in the second window,
    b_c, c_d, and b_d appear. This example is in the
    doctest below.
    Note that the order of the tokens in the feature name
    matches the order in which they appear in the document.
    (e.g., a__b, not b__a)

    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_pair_features(np.array(['a', 'b', 'c', 'd']), feats)
    >>> sorted(feats.items())
    [('token_pair=a__b', 1), ('token_pair=a__c', 1), ('token_pair=b__c', 2), ('token_pair=b__d', 1), ('token_pair=c__d', 1)]
    """
    store = defaultdict(list)
    counter=0
    entries=0
    for j in range(len(tokens)-2):
        i=j;
        for  i in range(j,len(tokens)):
            if(entries<=k-1):
                store[counter].append(tokens[i]);
                entries=entries+1
            else:
                entries=0
                counter=counter+1;
                break;
    i=0;
    j=0;
    while(True):
        arr=store[j]
        for i in range(len(store[j])):
            for k in range(i+1,len(arr)):
                if feats['token_pair='+arr[i]+"__"+arr[k]]>=1:
                   value=feats['token_pair='+arr[i]+"__"+arr[k]]
                   feats['token_pair='+arr[i]+"__"+arr[k]]=value+1
                else:   
                     feats['token_pair='+arr[i]+"__"+arr[k]]=1
        j=j+1
        if(j>=len(store)):
            break
    
    ###TODO
    pass


neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])

def lexicon_features(tokens, feats):
    """
    Add features indicating how many time a token appears that matches either
    the neg_words or pos_words (defined above). The matching should ignore
    case.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    In this example, 'LOVE' and 'great' match the pos_words,
    and 'boring' matches the neg_words list.
    >>> feats = defaultdict(lambda: 0)
    >>> lexicon_features(np.array(['i', 'LOVE', 'this', 'great', 'boring', 'movie']), feats)
    >>> sorted(feats.items())
    [('neg_words', 1), ('pos_words', 2)]
    """
    feats["neg_words"]=0
    feats["pos_words"]=0
    for i in range(len(tokens)):
        if(tokens[i].lower() in neg_words  ):
            feats["neg_words"]=feats["neg_words"]+1
        if(tokens[i].lower() in pos_words  ):
            feats["pos_words"]=feats["pos_words"]+1
    ###TODO
    pass


def featurize(tokens, feature_fns):
    """
    Compute all features for a list of tokens from
    a single document.

    Params:
      tokens........array of token strings from a document.
      feature_fns...a list of functions, one per feature
    Returns:
      list of (feature, value) tuples, SORTED alphabetically
      by the feature name.

    >>> feats = featurize(np.array(['i', 'LOVE', 'this', 'great', 'movie']), [token_features, lexicon_features])
    >>> feats
    [('neg_words', 0), ('pos_words', 2), ('token=LOVE', 1), ('token=great', 1), ('token=i', 1), ('token=movie', 1), ('token=this', 1)]
    """
    feats = defaultdict(lambda: 0)
    for features in feature_fns:
        features(tokens,feats)
    #feature_fns[1](tokens,feats)
    feats= sorted(feats.items(),key=lambda x:(x[0]))
    return feats
    ###TODO
    pass


def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    """
    Given the tokens for a set of documents, create a sparse
    feature matrix, where each row represents a document, and
    each column represents a feature.

    Params:
      tokens_list...a list of lists; each sublist is an
                    array of token strings from a document.
      feature_fns...a list of functions, one per feature
      min_freq......Remove features that do not appear in
                    at least min_freq different documents.
    Returns:
      - a csr_matrix: See https://goo.gl/f5TiF1 for documentation.
      This is a sparse matrix (zero values are not stored).
      - vocab: a dict from feature name to column index. NOTE
      that the columns are sorted alphabetically (so, the feature
      "token=great" is column 0 and "token=horrible" is column 1
      because "great" < "horrible" alphabetically),

    >>> docs = ["Isn't this movie great?", "Horrible, horrible movie"]
    >>> tokens_list = [tokenize(d) for d in docs]
    >>> feature_fns = [token_features]
    >>> X, vocab = vectorize(tokens_list, feature_fns, min_freq=1)
    >>> type(X)
    <class 'scipy.sparse.csr.csr_matrix'>
    >>> X.toarray()
    array([[1, 0, 1, 1, 1, 1],
           [0, 2, 0, 1, 0, 0]], dtype=int64)
    >>> sorted(vocab.items(), key=lambda x: x[1])
    [('token=great', 0), ('token=horrible', 1), ('token=isn', 2), ('token=movie', 3), ('token=t', 4), ('token=this', 5)]
    """
   
    features=defaultdict(lambda: 0);
    CountinDoc=defaultdict(lambda: 0);
    dict_final=defaultdict(lambda: 0)
    rowcounter=0
    col_dict=defaultdict(lambda: 0)
    if(rowcounter==0):
        for d in tokens_list:
            features[rowcounter]= featurize(d,feature_fns)
            rowcounter=rowcounter+1
    else:
        while rowcounter<len(tokens_list):
            listvalue=tokens_list[rowcounter]
            features[rowcounter]= featurize(listvalue,feature_fns)
            rowcounter=rowcounter+1
            
    for key,value in features.items():

        for subvalue in value:
       
                if(CountinDoc[subvalue[0]]!=0):
                    CountinDoc[subvalue[0]]=CountinDoc[subvalue[0]]+1
                else:
                    CountinDoc[subvalue[0]]=1
    
    
    if(vocab==None):    
        for key,values in  CountinDoc.items():
            if(CountinDoc[key]>=min_freq):
                dict_final[key]=CountinDoc[key]
        dict_final= sorted(dict_final.items(),key=lambda x:(x[0]))
        counter=0
        for value in dict_final:
            col_dict[value[0]]=counter
            counter=counter+1
            
    data = []
    row = []
    column=[]
    counter=0
    colcounter=0
    flagvoacab=0
    if(vocab!=None):
        col_dict=vocab
        for key,value in features.items():
       # colcounter=0
            for term in value:
                if(term[0] in col_dict):
                #index = vocabulary.setdefault(term, len(vocabulary))
                    row.append(colcounter)
                    column.append(col_dict[term[0]])
                    data.append(term[1])
            colcounter=colcounter+1
            counter=counter+1
    else:    
    
       for key,value in features.items():
       # colcounter=0
         for term in value:
             if(CountinDoc[term[0]]>=min_freq):
                #index = vocabulary.setdefault(term, len(vocabulary))
                row.append(colcounter)
                column.append(col_dict[term[0]])
                data.append(term[1])
         colcounter=colcounter+1
         counter=counter+1
        #indptr.append(len(row))
    ###TODO
    return  csr_matrix((data, (row, column)), dtype='int64'),col_dict
    pass


def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.CountinDoc[values]
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    """
    Compute the average testing accuracy over k folds of cross-validation. You
    can use sklearn's KFold class here (no random seed, and no shuffling
    needed).

    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds. 

    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
    accuracies=[]
    
    if len(labels) < k:
        k=len(labels)
    kfoldvalue = KFold(len(labels), k)
    for trainidx, testidx in kfoldvalue:
        clf.fit(X[trainidx], labels[trainidx])
        accuracies.append(accuracy_score(labels[testidx], clf.predict(X[testidx])))
        #total=total + acc
        
    return np.mean(accuracies)
    
    ###TODO
    pass


def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    """
    Enumerate all possible classifier settings and compute the
    cross validation accuracy for each setting. We will use this
    to determine which setting has the best accuracy.

    For each setting, construct a LogisticRegression classifier
    and compute its cross-validation accuracy for that setting.

    In addition to looping over possible assignments to
    keep_internal_punct and min_freqs, we will enumerate all
    possible combinations of feature functions. So, if
    feature_fns = [token_features, token_pair_features, lexicon_features],
    then we will consider all 7 combinations of features (see Log.txt
    for more examples).

    Params:
      docs..........The list of original training documents.
      labels........The true labels for each training document (0 or 1)
      punct_vals....List of possible assignments to
                    keep_internal_punct (e.g., [True, False])
      feature_fns...List of possible feature functions to use
      min_freqs.....List of possible min_freq values to use
                    (e.g., [2,5,10])

    Returns:
      A list of dicts, one per combination. Each dict has
      four keys:
      'punct': True or False, the setting of keep_internal_punct
      'features': The list of functions used to compute features.
      'min_freq': The setting of the min_freq parameter.
      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.

      This list should be SORTED in descending order of accuracy.

      This function will take a bit longer to run (~20s for me).
    """
    
    dict_list=[]
    feature_fns_list=[]
    for val in range(1, len(feature_fns)+1):
        for item in combinations(feature_fns, val):
            feature_fns_list.append(item)

    
    tokens_list_true = [tokenize(d,True) for d in docs]
    tokens_list_false = [tokenize(d) for d in docs]

    #min_freqs=[2]
    for punct in punct_vals:
        for freq in min_freqs:
            
            for fns in feature_fns_list:
                feature_fns1=fns
            
                settings={}
                
                settings["features"]=feature_fns1
                settings["min_freq"]=freq
                settings["punct"]=punct               
                
                               
                if (punct==False):
                    X,vocab=vectorize(tokens_list_false, feature_fns1, freq)
                else:
                    X,vocab=vectorize(tokens_list_true, feature_fns1, freq)                    

                settings["accuracy"]=cross_validation_accuracy(LogisticRegression(), X, labels, 5)
                dict_list.append(settings)
  
    return  sorted(dict_list, key=lambda k: k["accuracy"],reverse=True) 
    ###TODO
    pass
    



def mean_accuracy_per_setting(results):
    """
    To determine how important each model setting is to overall accuracy,
    we'll compute the mean accuracy of all combinations with a particular
    setting. For example, compute the mean accuracy of all runs with
    min_freq=2.

    Params:
      results...The output of eval_all_combinations
    Returns:
      A list of (accuracy, setting) tuples, SORTED in
      descending order of accuracy.
    """

    listaccuracy=dict()
    settings=defaultdict(lambda: 0)

    listaccuracy["punct=True"]=0
    listaccuracy["punct=False"]=0

    counter=0
    functiondict=defaultdict(lambda: 0)
    functionslist=defaultdict(lambda: 0)
    strdict=" "
    dictvalue=dict()
    for index in range(0,(len(results))):
        functiontype=results[index]["features"]
        
        strdict="features="
        for t1 in functiontype:
            strdict=strdict+" "+t1.__name__
        if(settings[strdict]!=0):
            settings[strdict]=settings[strdict]+results[index]["accuracy"]
            listaccuracy[strdict]= listaccuracy[strdict]+1
        else:
            settings[strdict]=results[index]["accuracy"]
            listaccuracy[strdict]=1
     

    settings["punct=True"]=0
    settings["punct=False"]=0
   
    for index in range(0,(len(results))):
        typevalue=results[index]["punct"]
        freqvalue=results[index]["min_freq"]
       
       
       
       
       
       
        strdict="min_freq="+str(freqvalue)
        if(typevalue==True):
            settings["punct=True"]=settings["punct=True"]+results[index]["accuracy"]
            listaccuracy["punct=True"]=listaccuracy["punct=True"]+1
            #settings['True"]=0
        if(typevalue==False):
            settings["punct=False"]=settings["punct=False"]+results[index]["accuracy"]   
            listaccuracy["punct=False"]= listaccuracy["punct=False"]+1
        if(settings[strdict]!=0):
            settings[strdict]=settings[strdict]+results[index]["accuracy"]
            listaccuracy[strdict]=1+listaccuracy[strdict]
        else:    
            settings[strdict]=results[index]["accuracy"]
            listaccuracy[strdict]=1

    for  key,value in settings.items():
        settings[key]=value/listaccuracy[key]

    
    settings_rev=defaultdict(lambda: 0)
    for key,value in settings.items():
        settings_rev[value]=key
    return sorted(settings_rev.items(), key=lambda k: k[0],reverse=True)
    
    ###TODO
    pass


def fit_best_classifier(docs, labels, best_result):
    """
    Using the best setting from eval_all_combinations,
    re-vectorize all the training data and fit a
    LogisticRegression classifier to all training data.
    (i.e., no cross-validation done here)

    Params:
      docs..........List of training document strings.
      labels........The true labels for each training document (0 or 1)
      best_result...Element of eval_all_combinations
                    with highest accuracy
    Returns:
      clf.....A LogisticRegression classifier fit to all
            training data.
      vocab...The dict from feature name to column index.
    """
    
    
    clf=LogisticRegression()
    if(best_result["punct"]==True):
        tokens_list = [tokenize(d,True) for d in docs]
    else:    
        tokens_list = [tokenize(d) for d in docs]
    
    X,vocab=vectorize(tokens_list, best_result["features"], best_result["min_freq"])
   
    clf.fit(X, labels) 
    return clf,vocab
   
    
    ###TODO
    pass


def top_coefs(clf, label, n, vocab):
    """
    Find the n features with the highest coefficients in
    this classifier for this label.
    See the .coef_ attribute of LogisticRegression.

    Params:
      clf.....LogisticRegression classifier
      label...1 or 0; if 1, return the top coefficients
              for the positive class; else for negative.
      n.......The number of coefficients to return.
      vocab...Dict from feature name to column index.
    Returns:
      List of (feature_name, coefficient) tuples, SORTED
      in descending order of the coefficient for the
      given class label.
    """

    #Refrences:https://github.com/iit-cs579/main/blob/master/lec/l11/l11.ipynb
    top_coef_terms=[]
    coef = clf.coef_[0]
    listvalue=[]
  #  word_list = np.array([key for key,val in ]) 
    token_list=sorted(vocab.items(), key=lambda x:x[1])
    for val,key in token_list:
        listvalue.append(val)
    word_list=np.array(listvalue)
    if label==0:
        top_coef_ind = np.argsort(coef)[:n]
    else:
        top_coef_ind = np.argsort(coef)[::-1][:n]

    top_coef_terms=word_list[top_coef_ind]

    top_coef = abs(coef[top_coef_ind])

    final_result=[x for x in zip(top_coef_terms, top_coef)]
    return sorted(final_result, key=lambda x:x[1],reverse=True)
     ###TODO
    pass


def parse_test_data(best_result, vocab):
    """
    Using the vocabulary fit to the training data, read
    and vectorize the testing data. Note that vocab should
    be passed to the vectorize function to ensure the feature
    mapping is consistent from training to testing.

    Note: use read_data function defined above to read the
    test data.

    Params:
      best_result...Element of eval_all_combinations
                    with highest accuracy
      vocab.........dict from feature name to column index,
                    built from the training data.
    Returns:
      test_docs.....List of strings, one per testing document,
                    containing the raw.
      test_labels...List of ints, one per testing document,
                    1 for positive, 0 for negative.
      X_test........A csr_matrix representing the features
                    in the test data. Each row is a document,
                    each column is a feature.
    """
    docs = read_data_test(os.path.join('data'))
    if(best_result["punct"]==True):
        tokens_list = [tokenize(d,True) for d in docs]
    else:    
        tokens_list = [tokenize(d) for d in docs]
    X,vocab=vectorize(tokens_list, best_result["features"], best_result["min_freq"],vocab=vocab)
    return  docs,X
    ###TODO
    pass


def print_top_misclassified(test_docs, test_labels, X_test, clf, n,utf):
    """
    Print the n testing documents that are misclassified by the
    largest margin. By using the .predict_proba function of
    LogisticRegression <https://goo.gl/4WXbYA>, we can get the
    predicted probabilities of each class for each instance.
    We will first identify all incorrectly classified documents,
    then sort them in descending order of the predicted probability
    for the incorrect class.
    E.g., if document i is misclassified as positive, we will
    consider the probability of the positive class when sorting.

    Params:
      test_docs.....List of strings, one per test document
      test_labels...Array of true testing labels
      X_test........csr_matrix for test data
      clf...........LogisticRegression classifier fit on all training
                    data.
      n.............The number of documents to print.

    Returns:
      Nothing; see Log.txt for example printed output.
    """
    
    predicted_label = clf.predict(X_test)
    pos=0
    neg=0
    print("\n Number of instances per class found:")
    utf.write("\n Number of instances per class found:")
    for labels in predicted_label:
        if(labels==1):
            pos=pos+1
        else:
            neg=neg+1
    print("\n Positive="+str(pos))
    utf.write("\n Positive="+str(pos))
    utf.write("\n Negative="+str(neg))
    print("\n Negative="+str(neg))
    probablities_label = clf.predict_proba(X_test)
    dictlist = []
    flag=1
    i=0
    j=i
    length0ftest_labels=len(test_docs)
    while(i<length0ftest_labels):
                dictvalues = {}
        # if ( dictvalues["post"] < probablities_label[i][0]  and flag==1):
                dictvalues["filename"]= test_docs[i]
                dictvalues["truth"] =test_labels[i]
                dictvalues["negprobas"]=probablities_label[i][1]
                dictvalues["probas"]=probablities_label[i][0]
                dictvalues["sentiments"]=predicted_label[i]
                dictlist.append(dictvalues)
        
                i=i+1
    print("Top  5 Positive Comments\n")
    utf.write("\nTop 5 Positive Comments\n")
    i=0
    dict_list = sorted(dictlist, key=lambda x: x["probas"],reverse=True)    
    for item in  dict_list:
        if(i==5):
            break
        if(item["sentiments"]==1):
         utf.write(item["filename"]+ "\n") 
         print(item["filename"]+ "\n")
         i=i+1
    utf.write("Top 5 Negative Comments\n")
    print("Top 5 Negative Comments\n")
    dict_list = sorted(dictlist, key=lambda x: x["negprobas"])   
    i=0
    for item in  dict_list:
        if(i==5):
            break
        if(item["sentiments"]==0):
         print(item["filename"]+"\n")        
         utf.write(item["filename"]+ "\n") 
         i=i+1
    ###TODO
    pass

def replace_with_not(string):
    ns = ""
    suffix = "n't"
    word = "not"
    for stn in string.split():
        if stn.endswith(suffix):
            stn = stn[:-3]
            ns += stn + ' ' + word + ' '
        else:
            ns += stn + ' '
    return ns


def read_create_test(utf):
    
    '''
    Reads Trump.csv and create a folder for Test data
    Parameters:
    utf:File object to  write in file
    '''
    newpath = r'data\test' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    df = pd.read_pickle("Trump.csv")
    list1=dict()
    index=0;
    count=0
#    stemmer2 = SnowballStemmer("english", ignore_stopwords=True)
    for values in df["message"]:
        t1=values
        score=0.0
        values=values.replace('_', '')
        strnewvals=replace_with_not(values)
        if(strnewvals!=""):
            values=strnewvals
        tokens=values
        for t in tokens:
            for vals in t:
                if(ord(vals)>132 or ord(vals)<0):
                    values=values.replace(vals,"");
        file = open("data\\test\\"+df["Id"][index]+".txt",'w')  
          #  print(index) 
        file.write(values)
        index=index+1

    utf.write("\nTotal FB comments for Test DataSet="+str(index))
    #
def main():
    """
    Put it all together.
    ALREADY DONE.
    """
    with zipfile.ZipFile('train.zip',"r") as z:
       z.extractall("data")
    utf=io.open('classify.txt', 'w', encoding='utf8')
    utf.close()
    utf=io.open('classify.txt', 'a', encoding='utf8')
    read_create_test(utf)
    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Download and read data.
    #download_data()
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    results = eval_all_combinations(docs, labels,
                                    [True],
                                    feature_fns,
                                [2,5,10])
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    utf.write('\nbest cross-validation result:\n%s' % str(best_result))
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    utf.write('worst cross-validation result:\n%s' % str(worst_result))
    utf.close()
    utf=io.open('classify.txt', 'a', encoding='utf8')
    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    utf.write('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    utf.write('\n negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    utf.write('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    utf.write('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))
    utf.write('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))

    # Parse test data
    test_docs,  X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
 #   print('testing accuracy=%f' %
  #        accuracy_score(test_labels, predictions))

   
    print_top_misclassified(test_docs, labels, X_test, clf, 5,utf)
    utf.close()

if __name__ == '__main__':
    main()
