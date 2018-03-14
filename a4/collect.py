
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import os

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

consumer_key = 'keqAjGarsPukpLSwlkesi9H3R'
consumer_secret = '0x2xrx5opDRXK2LXcWlR38ALT7i2GvGA58hKP8ZzZIHKnEKcqc'
access_token = '1299787945-bd48GieE8Gw6sfG2pdvuOPPNWdlBnHcqrrZl3wK'
access_token_secret = 'qipdICMzCjIUC9FJ7N7s9ga8hHo4bIyW0KPPz4QQStgvX'

def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)




def collect(twitter):
    tweets = []
    retweets=[]
    
    if not os.path.isfile("tweets.txt"):
        while(len(tweets)<10):
            for r in robust_request(twitter,'search/tweets',{'q': 'feminism -filter:retweets', 'count': 100,'lang':'en'}):
                if r not in tweets:
                    tweets.append(r)
            #print(len(tweets))
        
        pickle.dump(tweets,open("tweets.txt","wb")) # write the tweets in a pickle file called tweets.txt
        
        tweets_text=[]
        user_screen_name=[]
        user_id=[]
        #print(len(tweets))
        for i in range(len(tweets)):
            tweets_text.append(tweets[i]['text'])
            user_screen_name.append(tweets[i]['user']['screen_name'])
            user_id.append(tweets[i]['user']['id'])
        
        pickle.dump(tweets_text,open("tweets_text.txt","wb")) # write the tweets_text in a pickle file called tweets_text.txt
        
        pickle.dump(user_screen_name,open("user_screen_names.txt","wb")) 
        
        pickle.dump(user_id,open("user_id.txt","wb")) 
        
        url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
        zipfile = ZipFile(BytesIO(url.read()))
        afinn_file = zipfile.open('AFINN/AFINN-111.txt')
        
        afinn = dict()
        
        pos_words=[]
        neg_words=[]
        
        
        for line in afinn_file:
            parts = line.strip().split()
            if len(parts) == 2:
                afinn[parts[0].decode("utf-8")] = int(parts[1])

        
        for k,v in afinn.items():
            if v<0:
                neg_words.append(k)
            if v>0:
                pos_words.append(k)
        
        
        pickle.dump(pos_words,open("pos_words.txt","wb")) 
        
        pickle.dump(neg_words,open("neg_words.txt","wb")) 
        
        unique_screen_names=[]
        for name in user_screen_name:
            if name not in unique_screen_names:
                unique_screen_names.append(name)
        
        '''
        friends_list=[]
        for r in robust_request(twitter,'friends/ids',{'screen_name': unique_screen_names[0]}):
            friends_list.append(r)
        
        print(friends_list)
        print(len(friends_list))
        '''
        friends_dict = defaultdict(dict)
        for name in unique_screen_names:
            friends_id=[]
            for r in robust_request(twitter,'friends/ids',{'screen_name': name,'count':10}):
                friends_id.append(r)
            friends_dict[name]=friends_id
            #print(len(friends_id))
        
        
        pickle.dump(friends_dict,open("friends_followers_ids.txt","wb")) 

    pass


def main():
    twitter =  TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)
    
    collect(twitter)
if __name__ == '__main__':
    main()
    