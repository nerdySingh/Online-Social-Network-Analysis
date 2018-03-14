from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import string
import pickle
import time
from TwitterAPI import TwitterAPI
import numpy as np
from numpy import array
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import numpy.lib.arraysetops as aso
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen


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
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])

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
    """
    ###TODO
    l1=[]
    if keep_internal_punct == False:
        val=re.sub('[^a-zA-Z0-9_]+', ' ',doc.lower()).split()
        
        arr1= np.array(val)
        return arr1
    else:
        for x in doc.lower().split():
            val=re.sub('[^a-zA-Z0-9_*-]+$','',re.sub('^[^a-zA-Z0-9_*-]+','',x)) 
            l1.append(val)
        return np.array(l1)
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
    ###TODO
    new_tokens =[]
    for i in range(len(tokens)):
        new_tokens.append("token="+tokens[i])
    feats1 = Counter(new_tokens)
    for k,v in feats1.items():
        feats[k]=v
    sorted(feats.items())
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
    ###TODO
    token_w=[]
    pair_t=[]
    new_l =[]
    val = tokens.tolist()
    for i in range(len(val)):
        token_w.append(val[i:k+i])
    for vv in token_w:
        if len(vv)==k:
            new_l.append(vv)
        #print(vv)
    
    for i in range(len(new_l)):
        for j in range(len(new_l[i])):
            for k in range(j+1,len(new_l[i])):
                    pair ="token_pair="+new_l[i][j]+"__"+new_l[i][k]
                    pair_t.append(pair)
    feat1= Counter(pair_t)
    for k,v in feat1.items():
        feats[k]=v
    return sorted(feats.items())

    pass


neg_words=set(pickle.load(open("neg_words.txt","rb")))
pos_words=set(pickle.load(open("pos_words.txt","rb")))

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
    ###TODO
    count1=0
    count2=0
    
    for vv in tokens:
        val=vv.lower()
        #print(vv.lower())
        if val in neg_words:
            #print (vv)
            count1+=1
        elif val in pos_words:
            #print (vv+"sdaa")
            count2+=1
    
    feats['neg_words'] = count1
    feats['pos_words'] = count2
    
    
    sorted(feats.items())
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
    ###TODO
    val=[]
    feats = defaultdict(lambda:0)
    
    for f in feature_fns:
        f(tokens,feats)
                
         
    return sorted(list(feats.items()))
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
    ###TODO
    list_n=[]
    col_data=[]
    col_data1=[]
    feats = defaultdict(lambda:0)
    for i in range(len(tokens_list)):
        feats=featurize(np.array(tokens_list[i]),feature_fns)
        list_n.append(dict(feats))
    
    if vocab ==None:
        mydict1 = defaultdict(lambda:0)
        mydict2=defaultdict(lambda:0)
        myval=[]
        
        for dic in list_n:
            for k,v in dic.items():
                if dic[k]>0:
                    mydict1[k]=mydict1[k]+1
                if (k not in mydict2) and (mydict1[k]>=min_freq):
                    myval.append(k)
                    mydict2[k]=0
        myval =  sorted(myval)
        f=0
        vocab = defaultdict(lambda:0)
        for m in myval:
            vocab[m]=f
            f+=1
        row=[]
        column=[]
        data=[]
        
        counter=0
        for dic in list_n:
            for k,v in dic.items():
                if k in vocab:
                    row.append(counter)
                    column.append(vocab[k])
                    data.append(v)
            counter+=1
        
        x1 = np.array(data,dtype='int64')
        x2= np.array(row,dtype='int64')
        x3=np.array(column,dtype='int64')

        mat1=csr_matrix( (x1,(x2,x3)), shape=(counter,len(vocab)) )
        return mat1 , vocab
    
    else:
          row=[]
          column=[]
          data=[]
          counter=0
          for dic in list_n:
                for k,v in dic.items():
                      if k in vocab:
                            row.append(counter)
                            column.append(vocab[k])
                            data.append(v)
                counter+=1
        
          x1 = np.array(data,dtype='int64')
          x2= np.array(row,dtype='int64')
          x3=np.array(column,dtype='int64')

          mat1=csr_matrix( (x1,(x2,x3)), shape=(counter,len(vocab)) )
          return mat1 , vocab
    
    
    pass


def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
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
    ###TODO
    model =clf
    le = len(X.toarray())
    cv = KFold(len(labels),k)
    
    accuracies =[]
    Y =labels
    for train_ind ,test_ind in cv:
          model.fit(X[train_ind],Y[train_ind])
          predictions = model.predict(X[test_ind])
          accuracies.append(accuracy_score(Y[test_ind],predictions))
    return np.mean(accuracies)
    pass


def eval_all_combinations(docs, labels, punct_vals,feature_fns, min_freqs):
    data_true=[]
    data_false=[]
    X = docs
    Y = labels
    k = len(docs)
    final_feat={}
    model = LogisticRegression()
    for val in punct_vals:
        for i in range(len(docs)):
            if val == False:
                text= tokenize(docs[i],val)
                data_false.append(text)
            else:
                text= tokenize(docs[i],val)
                data_true.append(text)
    #print(data_false)
    feat_pair1=[]
    for i in range(len(feature_fns)):
          feat_pair1.append(feature_fns[i:i+1])
    #print(feat_pair1)
    feat_pair2 = list(combinations(feature_fns,2))
    #print(feat_pair2[0])
    feat_pair3 = list(combinations(feature_fns,3))
    final_list=[]
    
    for val in punct_vals:
          for i in range(len(min_freqs)):
                my_dict_pair={}
                for j in range(len(feat_pair1)):
                      if val==False:
                            mat1,vocab = vectorize(data_false,feat_pair1[j],min_freqs[i])
                            if mat1!=None and vocab!=None:
                                  accu = cross_validation_accuracy(model,mat1,labels,5)
                                  my_dict_pair['features']=tuple(feat_pair1[j])
                                  my_dict_pair['punct'] = val
                                  my_dict_pair['accuracy'] = accu
                                  my_dict_pair['min_freq']=min_freqs[i]
                                  final_list.append(my_dict_pair)
                      else:
                            mat1,vocab = vectorize(data_true,feat_pair1[j],min_freqs[i])
                            if mat1!=None and vocab!=None:
                                  accu = cross_validation_accuracy(model,mat1,labels,5)
                                  my_dict_pair['features']=tuple(feat_pair1[j])
                                  my_dict_pair['punct'] = val
                                  my_dict_pair['accuracy'] = accu
                                  my_dict_pair['min_freq']=min_freqs[i]
                                  final_list.append(my_dict_pair)
    
    
    
    
    
    
    
    for val in punct_vals:
          for i in range(len(min_freqs)):
                my_dict_pair={}
                for j in range(len(feat_pair2)):
                      if val==False:
                            mat1,vocab = vectorize(data_false,list(feat_pair2[j]),min_freqs[i])
                            if mat1!=None and vocab!=None:
                                  accu = cross_validation_accuracy(model,mat1,labels,5)
                                  my_dict_pair['features']=feat_pair2[j]
                                  my_dict_pair['punct'] = val
                                  my_dict_pair['accuracy'] = accu
                                  my_dict_pair['min_freq']=min_freqs[i]
                                  final_list.append(my_dict_pair)
                      else:
                            mat1,vocab = vectorize(data_true,list(feat_pair2[j]),min_freqs[i])
                            if mat1!=None and vocab!=None:
                                  accu = cross_validation_accuracy(model,mat1,labels,5)
                                  my_dict_pair['features']=feat_pair2[j]
                                  my_dict_pair['punct'] = val
                                  my_dict_pair['accuracy'] = accu
                                  my_dict_pair['min_freq']=min_freqs[i]
                                  final_list.append(my_dict_pair)

    for val in punct_vals:
          for i in range(len(min_freqs)):
                my_dict_pair={}
                for j in range(len(feat_pair3)):
                      if val==False:
                            mat1,vocab = vectorize(data_false,list(feat_pair3[j]),min_freqs[i])
                            if mat1!=None and vocab!=None:
                                  accu = cross_validation_accuracy(model,mat1,labels,5)
                                  my_dict_pair['features']=feat_pair3[j]
                                  my_dict_pair['punct'] = val
                                  my_dict_pair['accuracy'] = accu
                                  my_dict_pair['min_freq']=min_freqs[i]
                                  final_list.append(my_dict_pair)
                      else:
                            mat1,vocab = vectorize(data_true,list(feat_pair3[j]),min_freqs[i])
                            if mat1!=None and vocab!=None:
                                  accu = cross_validation_accuracy(model,mat1,labels,5)
                                  my_dict_pair['features']=feat_pair3[j]
                                  my_dict_pair['punct'] = val
                                  my_dict_pair['accuracy'] = accu
                                  my_dict_pair['min_freq']=min_freqs[i]
                                  final_list.append(my_dict_pair)
    
    return sorted(final_list,key=lambda x:(-x['accuracy'],-x['min_freq']))

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
    ###TODO
    punct=0
    min_freq=0
    feat_fn=0
    len1= len(docs)
    for k,v in best_result.items():
          if k=='punct':
                punct=best_result.get(k)
          if k=='min_freq':
                min_freq = best_result.get(k)
          if k=="features":
                feat_fn=best_result.get(k)
          
    data=[]
    for i in range(len(docs)):
          val=tokenize(docs[i],punct)
          data.append(val)
    
    mat1,vocab = vectorize(data,list(feat_fn),min_freq)
    clf = LogisticRegression()
    clf.fit(mat1[:len1],labels)

    return clf,vocab

    
    
    pass


def get_unique_tweets(tweets,best_result,vocab):
    unique_tweets=[]
    test_tokens=[]
    for x in tweets:
        if x not in unique_tweets:
            unique_tweets.append(x)
    
    
    
    for val in unique_tweets:
        data1=tokenize(val,best_result['punct'])
        test_tokens.append(data1)
    
    #print(best_result['min_freq'])
    X_test,vocab1 = vectorize(test_tokens,list(best_result['features']),best_result['min_freq'],vocab)
    #print(X_test)
    return X_test, vocab1, test_tokens
    


def my_predict(clf,X_test,tweets1):
    
    neg_tweets=[]
    pos_tweets=[]
    tweets=[]

    for x in tweets1:
        if x not in tweets:
            tweets.append(x)




    predict = clf.predict(X_test)
    for i, val in enumerate(predict):
        if val == 0:
            neg_tweets.append(tweets[i])
        if val==1:
            pos_tweets.append(tweets[i])
    return neg_tweets,pos_tweets

def classify():
    feature_fns = [token_features, token_pair_features, lexicon_features]
    docs, labels = read_data(os.path.join('data', 'train'))
    #print(docs[0])
    tweets= pickle.load(open("tweets_text.txt","rb"))
    tweets1=tweets
    results = eval_all_combinations(docs, labels,[True,False],feature_fns,[2,5,10])
    best_result = results[0]
    clf, vocab = fit_best_classifier(docs, labels, results[0])
    X_test,vocab,tweets = get_unique_tweets(tweets,best_result,vocab)
    #print(clf)
    neg_tweets,pos_tweets=my_predict(clf,X_test,tweets1)
    
    pickle.dump(neg_tweets,open("final_neg_tweets.txt","wb"))
    pickle.dump(pos_tweets,open("final_pos_tweets.txt","wb"))
    #print("Negative Tweets",neg_tweets[:5])
    #print("Positive Tweets",pos_tweets[:5])
    pass

def main():
    classify()
    pass


if __name__ == "__main__":
    main()

