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
    ###TODO
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


def plot_sorted_accuracies(results):
    """
    Plot all accuracies from the result of eval_all_combinations
    in ascending order of accuracy.
    Save to "accuracies.png".
    """
    ###TODO
    plot_list=[]
    for i in range(len(results)):
          for k,v in results[i].items():
                if k == "accuracy":
                      plot_list.append(v)
    
    plot_list = sorted(plot_list)
    plt.plot(plot_list)
    plt.ylabel('accuracy')
    plt.xlabel('setting')
    plt.savefig("accuracies.png")
    plt.show()
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

    ###TODO
    final_list1=[]
    feat_pair_1=[]
    check_l=[]
    min_freq_2=[]
    min_freq_5=[]
    min_freq_10=[]
    punct_true=[]
    punct_false=[]
    feature_name1=[]
    feature_val1=[]
    feature_name2=[]
    feature_val2=[]
    feature_name3=[]
    feature_val3=[]

    for i in range(len(results)):
          for k,v in results[i].items():
                if k=="min_freq":
                      if v==2:
                            val = results[i]['accuracy']
                            min_freq_2.append(val)
                      if v==5:
                            val = results[i]['accuracy']
                            min_freq_5.append(val)
                      if v==10:
                            val = results[i]['accuracy']
                            min_freq_10.append(val)
                
                if k=="punct":
                      
                      if v==True:
                            val = results[i]['accuracy']
                            punct_true.append(val)

                      if v==False:
                            val = results[i]['accuracy']
                            punct_false.append(val)
                
                if k=="features":
                      if len(v) ==1:
                            if v[0].__name__ not in feature_name1:
                                  feature_name1.append(v[0].__name__)
                            if v not in feat_pair_1:
                                  feat_pair_1.append(v)
                                  feature_val1.append(results[i]['accuracy'])
                      if len(v) ==2:
                            test_val=[]
                            for j in range(len(v)):
                                  test_val.append(v[j].__name__)
                            feature_val2.append(results[i]['accuracy'])
                            if test_val not in feature_name2:
                                  feature_name2.append(test_val)
                            


                      if len(v)==3:
                            test_val=[]
                            for j in range(len(v)):
                                  test_val.append(v[j].__name__)
                            feature_val3.append(results[i]['accuracy'])
                            if test_val not in feature_name3:
                                  feature_name3.append(test_val)
    
    
    
    result3=[]
    result2=[]
    counter=0
    ki=0
    
    

    for i in range(len(feat_pair_1)):
          mydict1={}
          mydict1[np.mean(feature_val1[i])]="feature="+feature_name1[i]
          final_list1.append(mydict1)              

    for i in range(len(feature_name3)):
      val=""
      mydict1={}
      for j in range(len(feature_name3[i])):
            val=val+" "+feature_name3[i][j]
      mydict1[np.mean(feature_val3[i])] = "feature="+val
      final_list1.append(mydict1)

    
    for i in range(len(feature_name2)):
      val=""
      mydict1={}
      for j in range(len(feature_name2[i])):
            val=val+" "+feature_name2[i][j]
      mydict1[np.mean(feature_val2[i])] = "feature="+val
      final_list1.append(mydict1)

    mydict1[np.mean(min_freq_2)]="min_freq=2"
    final_list1.append(mydict1)
    mydict1[np.mean(min_freq_5)]="min_freq=5"
    final_list1.append(mydict1)
    mydict1[np.mean(min_freq_10)]="min_freq=10"
    final_list1.append(mydict1)

    
    final_list3=[]
    final_list4=[]
    final_list2=[]
    for i in range(len(final_list1)):
          final_list2.append(list(final_list1[i].items()))
    
    for i in range(len(final_list2)):
          final_list3.append(final_list2[i])
    
    for i in range(len(final_list3)):
          for j in range(len(final_list3[i])):
                if final_list3[i][j] not in final_list4:
                      final_list4.append(final_list3[i][j])
    

    return sorted(final_list4, key=lambda x:-x[0])       
    
                            
                            
                            
    
    
      
                
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
    ###TODO
    if label ==0:
          coef = clf.coef_[label]
          final_list=[]
          top_coef_terms=[]
          top_coef_ind = np.argsort(coef)[:n]
          for i in top_coef_ind:
                for k,v in vocab.items():
                      if v==i:
                            top_coef_terms.append(k)
          top_coef = coef[top_coef_ind]
          for i in range(len(top_coef)):
                if top_coef[i]<1:
                      top_coef[i] = top_coef[i] * -1
          for i in range(len(top_coef_terms)):
                final_list.append((top_coef_terms[i],top_coef[i]))
    else:
          coef = clf.coef_[0]
          final_list=[]
          top_coef_terms=[]
          top_coef_ind = np.argsort(coef)[::-1][:n]
          for i in top_coef_ind:
                for k,v in vocab.items():
                      if v==i:
                            top_coef_terms.append(k)
          top_coef = coef[top_coef_ind]
          for i in range(len(top_coef_terms)):
                final_list.append((top_coef_terms[i],top_coef[i]))
          
    return final_list
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
    ###TODO
    
    test_docs, test_labels = read_data(os.path.join('data', 'test'))
    tokens_l=[]
    for val in test_docs:
          data1 = tokenize(val,best_result['punct'])
          tokens_l.append(data1)
    X_test,vocab1=vectorize(tokens_l,list(best_result['features']),best_result['min_freq'],vocab)
    
   

    return test_docs,test_labels,X_test
          
    pass


def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
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
    ###TODO
    list_n=[]
    
    predict=clf.predict(X_test)
    prob=clf.predict_proba(X_test)
    truth=[]
    for i in range(len(predict)):
          mydict={}
          if predict[i] != test_labels[i]:
                if predict[i] == 0:
                      mydict['truth'] = test_labels[i]
                      mydict['predicted'] = predict[i]
                      mydict['proba'] = prob[i][0]
                      mydict['data']= test_docs[i]
                else:
                      mydict['truth'] = test_labels[i]
                      mydict['predicted'] = predict[i]
                      mydict['proba'] = prob[i][1]
                      mydict['data']= test_docs[i]
                list_n.append(mydict)
    list_n = sorted(list_n,key=lambda x:(-x['proba']))
    list_n = list_n[:n]
    for val in list_n:
          print('truth=%d predicted =%d proba=%.4f'%(val['truth'],val['predicted'],val['proba']))
          print(val['data'],end='\n')
    #print(list_n)     
    '''
    
    for i in range(len(data)):
      mydict['truth'] = truth[i]
      mydict['predicted'] = predicted[i]
      mydict['proba'] = prob[i]
      mydict['data']= data[i]
      list_n.append(mydict)
    list_n = sorted(list_n,key=lambda x:(-x['proba']))
    list_n = list_n[:n]
    for val in list_n:
          print('truth=%d predicted =%d proba=%.4f'%(val['truth'],val[predicted],val['proba']))
          print(val['data'],end='\n')

    '''  
          
    pass

def main():
    """
    Put it all together.
    ALREADY DONE.
    """
    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Download and read data.
    download_data()
    print(os.path.join('data', 'train'))
    docs, labels = read_data(os.path.join('data', 'train'))
    print(labels)
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    '''
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))

    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))

    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %
          accuracy_score(test_labels, predictions))

    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)
    '''

if __name__ == '__main__':
    main()

