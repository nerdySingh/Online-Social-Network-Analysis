# coding: utf-8

"""
CS579: Assignment 0
Collecting a political social network

In this assignment, I've given you a list of Twitter accounts of 4
U.S. presedential candidates from the previous election.

The goal is to use the Twitter API to construct a social network of these
accounts. We will then use the [networkx](http://networkx.github.io/) library
to plot these links, as well as print some statistics of the resulting graph.

1. Create an account on [twitter.com](http://twitter.com).
2. Generate authentication tokens by following the instructions [here](https://dev.twitter.com/docs/auth/tokens-devtwittercom).
3. Add your tokens to the key/token variables below. (API Key == Consumer Key)
4. Be sure you've installed the Python modules
[networkx](http://networkx.github.io/) and
[TwitterAPI](https://github.com/geduldig/TwitterAPI). Assuming you've already
installed [pip](http://pip.readthedocs.org/en/latest/installing.html), you can
do this with `pip install networkx TwitterAPI`.

OK, now you're ready to start collecting some data!

I've provided a partial implementation below. Your job is to complete the
code where indicated.  You need to modify the 10 methods indicated by
#TODO.

Your output should match the sample provided in Log.txt.
"""

# Imports you'll need.
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI
import numpy as np
from numpy import array
import numpy.lib.arraysetops as aso
#from more_itertools import unique_everseen
consumer_key = 'keqAjGarsPukpLSwlkesi9H3R'
consumer_secret = '0x2xrx5opDRXK2LXcWlR38ALT7i2GvGA58hKP8ZzZIHKnEKcqc'
access_token = '1299787945-bd48GieE8Gw6sfG2pdvuOPPNWdlBnHcqrrZl3wK'
access_token_secret = 'qipdICMzCjIUC9FJ7N7s9ga8hHo4bIyW0KPPz4QQStgvX'


# This method is done for you.
def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


def read_screen_names(filename):
    file = open(filename,'r')
    screen_names = file.readlines()
    screen_names = [x.strip() for x in screen_names]
    return screen_names


# I've provided the method below to handle Twitter's rate limiting.
# You should call this method whenever you need to access the Twitter API.
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


def get_users(twitter, screen_names):
    """Retrieve the Twitter user objects for each screen_name.
    Params:
        twitter........The TwitterAPI object.
        screen_names...A list of strings, one per screen_name
    Returns:
        A list of dicts, one per user, containing all the user information
        (e.g., screen_name, id, location, etc)

    See the API documentation here: https://dev.twitter.com/rest/reference/get/users/lookup

    In this example, I test retrieving two users: twitterapi and twitter.

    >>> twitter = get_twitter()
    >>> users = get_users(twitter, ['twitterapi', 'twitter'])
    >>> [u['id'] for u in users]
    [6253282, 783214]
    """
    ###TODO
    users = []
    for name in screen_names:
        i=0
        request = twitter.request('users/lookup', {'screen_name' : name})
        user_data = [x for x in request]
        users.append(user_data[i])
        i = i+1 
    return users
 


def get_friends(twitter, screen_name):
    """ Return a list of Twitter IDs for users that this person follows, up to 5000.
    See https://dev.twitter.com/rest/reference/get/friends/ids

    Note, because of rate limits, it's best to test this method for one candidate before trying
    on all candidates.

    Args:
        twitter.......The TwitterAPI object
        screen_name... a string of a Twitter screen name
    Returns:
        A list of ints, one per friend ID, sorted in ascending order.

    Note: If a user follows more than 5000 accounts, we will limit ourselves to
    the first 5000 accounts returned.

    In this test case, I return the first 5 accounts that I follow.
    >>> twitter = get_twitter()
    >>> get_friends(twitter, 'aronwc')[:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    ###TODO
    #request =  twitter.request('friends/ids',{'screen_name':screen_name, 'count':5000})
    request = robust_request(twitter,'friends/ids',{'screen_name':screen_name, 'count':5000})
    if request != None:
        friends = [x for x in request]
        return friends
    else:
        print('Got error %s \nsleeping for 15 minutes.' % request.text)
    


def add_all_friends(twitter, users):
    """ Get the list of accounts each user follows.
    I.e., call the get_friends method for all 4 candidates.

    Store the result in each user's dict using a new key called 'friends'.

    Args:
        twitter...The TwitterAPI object.
        users.....The list of user dicts.
    Returns:
        Nothing

    >>> twitter = get_twitter()
    >>> users = [{'screen_name': 'aronwc'}]
    >>> add_all_friends(twitter, users)
    >>> users[0]['friends'][:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    ###TODO
    #friends = []
    for i in range(len(users)):
        data = get_friends(twitter, users[i]['screen_name'])
        users[i]['friends'] = data
    pass


def print_num_friends(users):
    """Print the number of friends per candidate, sorted by candidate name.
    See Log.txt for an example.
    Args:
        users....The list of user dicts.
    Returns:
        Nothing
    """
    ###TODO
    data = sorted(users, key=lambda x: x['screen_name'])
    for i in range(len(data)):
        print(data[i]['screen_name'],len(data[i]['friends']),end='\n')

    pass


def count_friends(users):
    """ Count how often each friend is followed.
    Args:
        users: a list of user dicts
    Returns:
        a Counter object mapping each friend to the number of candidates who follow them.
        Counter documentation: https://docs.python.org/dev/library/collections.html#collections.Counter

    In this example, friend '2' is followed by three different users.
    >>> c = count_friends([{'friends': [1,2]}, {'friends': [2,3]}, {'friends': [2,3]}])
    >>> c.most_common()
    [(2, 3), (3, 2), (1, 1)]
    """
    ###TODO
    c = Counter()
    for i in range(len(users)):
        c.update(users[i]['friends'])
    #print(c.most_common(len(users)))    
    return c
    pass


def friend_overlap(users):
    
    """
    Compute the number of shared accounts followed by each pair of users.

    Args:
        users...The list of user dicts.

    Return: A list of tuples containing (user1, user2, N), where N is the
        number of accounts that both user1 and user2 follow.  This list should
        be sorted in descending order of N. Ties are broken first by user1's
        screen_name, then by user2's screen_name (sorted in ascending
        alphabetical order). See Python's builtin sorted method.

    In this example, users 'a' and 'c' follow the same 3 accounts:
    >>> friend_overlap([
    ...     {'screen_name': 'a', 'friends': ['1', '2', '3']},
    ...     {'screen_name': 'b', 'friends': ['2', '3', '4']},
    ...     {'screen_name': 'c', 'friends': ['1', '2', '3']},
    ...     ])
    [('a', 'c', 3), ('a', 'b', 2), ('b', 'c', 2)]
    """
    ###TODO
    friends_list = []
    count_list =[]
    count =[]
    for i in range(len(users)):
        friends_list.append(users[i]['friends'])
    
    for i in range(len(friends_list)):
        for j in range(len(friends_list)):
            counts = 0
            if i != j:
                arr1 = np.array(friends_list[i])
                arr2 = np.array(friends_list[j])
                my_bool= np.array(np.in1d(arr1,arr2))
                for x in np.nditer(my_bool):
                    if x == True:
                        counts += 1
                count.append(counts)
    #count = sorted(count, key=lambda x : x[2], reverse=True)
    #print(count)
    count_new = []
    #count = sorted(count, key=lambda x : x, reverse=True)
    for i in range(len(count)):
        for j in range(i+1,len(count)):
            if count[i]== count[j]:
                count_new.append(count[j])
    #print(count_new)
    
    names_new = []
    for i in range(len(users)):
        for j in range(i+1,len(users)):
            if users[i]['screen_name'] != users[j]['screen_name']:
                names_new.append([users[i]['screen_name'],users[j]['screen_name']])

    #print(len(names_new))
    #print(len(count_new))
    for i in range(len(count_new)):
        names_new[i].append(count_new[i])
    names_new=sorted(names_new, key=lambda x : x[2], reverse=True)
    final = []
    for i in range(len(names_new)):
        final.append(tuple(names_new[i]))

    return final
    

   

def followed_by_hillary_and_donald(users, twitter):
    """
    Find and return the screen_name of the one Twitter user followed by both Hillary
    Clinton and Donald Trump. You will need to use the TwitterAPI to convert
    the Twitter ID to a screen_name. See:
    https://dev.twitter.com/rest/reference/get/users/lookup

    Params:
        users.....The list of user dicts
        twitter...The Twitter API object
    Returns:
        A string containing the single Twitter screen_name of the user
        that is followed by both Hillary Clinton and Donald Trump.
    """
    ###TODO
    user_names = []
    for i in range(len(users)):
        user_names.append(users[i]['screen_name'])

    user_names = user_names[2:]
    friends_id = []
    user_obs = []
    for i in range(len(user_names)):
        request = robust_request(twitter,'friends/ids',{'screen_name':user_names[i]})
        if request != None:
            user_obs.append([[x for x in request]])
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
    arr1 = np.array(user_obs[0][0])
    arr2 = np.array(user_obs[1][0])
    mybool = np.array(np.intersect1d(arr1,arr2))
    mybool = mybool.tolist()
    request = robust_request(twitter,'users/lookup',{'user_id':mybool})
    if request != None:
        username = [u for u in request]
        for i in range(len(username)):
            userNameNew = username[i]['screen_name']
        return userNameNew

def create_graph(users, friend_counts):
    """ Create a networkx undirected Graph, adding each candidate and friend
        as a node.  Note: while all candidates should be added to the graph,
        only add friends to the graph if they are followed by more than one
        candidate. (This is to reduce clutter.)

        Each candidate in the Graph will be represented by their screen_name,
        while each friend will be represented by their user id.

    Args:
      users...........The list of user dicts.
      friend_counts...The Counter dict mapping each friend to the number of candidates that follow them.
    Returns:
      A networkx Graph
    """
    ###TODO
    c = friend_counts
    network_friends = []
    for k in c:
        if c[k] > 1:
            network_friends.append(k)
    arr2 = np.array(network_friends)
    users_new = [u for u in users]
    friends_list = []
    mybool1 = []
    
    for i in range(len(users)):
        friends_list.append(users[i]['friends'])
    for i in range(len(friends_list)):
        arr1 = np.array(friends_list[i])
        comp = np.array(np.intersect1d(arr1,arr2))
        mybool1.append(comp.tolist())
    graph = nx.Graph()
    for i in range(len(users)):
        for j in range(len(mybool1[i])):
            graph.add_edge(users[i]['screen_name'], mybool1[i][j])
    return graph
    pass


def draw_network(graph, users, filename):
    """
    Draw the network to a file. Only label the candidate nodes; the friend
    nodes should have no labels (to reduce clutter).

    Methods you'll need include networkx.draw_networkx, plt.figure, and plt.savefig.

    Your figure does not have to look exactly the same as mine, but try to
    make it look presentable.
    """
    ###TODO
    new_users = [u for u in users]
    user_names=[]
    for i in range(len(new_users)):
        user_names.append(new_users[i]['screen_name'])
    labels = {k:k for k in user_names}
    pos = nx.spring_layout(graph)
    d = nx.degree(graph)
    nx.draw_networkx_labels(graph,pos,labels,font_size = 8, font_color='k')
    nx.draw_networkx_nodes(graph,pos,nodelist=d.keys(),node_size = 10)
    nx.draw_networkx_edges(graph,pos,edgelist=None,alpha = 0.1)
    plt.axis('off')
    plt.savefig('network.png',dpi=(200))
    # plt.show()
    
    pass


def main():
    """ Main method. You should not modify this. """
    twitter = get_twitter()
    screen_names = read_screen_names('candidates.txt')
    print('Established Twitter connection.')
    print('Read screen names: %s' % screen_names)
    get_users(twitter,screen_names)
    users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name'])
    print('found %d users with screen_names %s' %
          (len(users), str([u['screen_name'] for u in users])))
    add_all_friends(twitter, users)
    print('Friends per candidate:')
    print_num_friends(users)
    count_friends(users)
    friend_counts = count_friends(users)
    print('Most common friends:\n%s' % str(friend_counts.most_common(5)))
    print('Friend Overlap:\n%s' % str(friend_overlap(users)))
    print('User followed by Hillary and Donald: %s' % followed_by_hillary_and_donald(users, twitter))
    graph = create_graph(users, friend_counts)
    print('graph has %s nodes and %s edges' % (len(graph.nodes()), len(graph.edges())))
    draw_network(graph, users, 'network.png')
    print('network drawn to network.png')
    

if __name__ == '__main__':
    main()

# That's it for now! This should give you an introduction to some of the data we'll study in this course.
