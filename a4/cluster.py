from collections import Counter, defaultdict, deque
import matplotlib.pyplot as plt
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


def create_graph(user_names,user_ids):
    graph = nx.Graph()
    
    for i in range(len(user_names)):
        for j in range(len(user_ids)):
            for ids in user_ids[j]:
                graph.add_edge(user_names[i],ids)
    
    labels = {k:k for k in user_names}
    pos = nx.spring_layout(graph)
    d = nx.degree(graph)
    nx.draw_networkx_labels(graph,pos,labels,font_size = 8, font_color='k')
    nx.draw_networkx_nodes(graph,pos,nodelist=d.keys(),node_size = 10)
    nx.draw_networkx_edges(graph,pos,edgelist=None,alpha = 0.1)
    #nx.draw(G,with_labels=False)
    
    return graph
    pass



def bottom_up(root, node2distances, node2num_paths, node2parents):
    results=[]
    credits=defaultdict(int)
    nodes=defaultdict(int)
    ss=defaultdict(int)
    node2paths=node2num_paths
    
    for k,v in node2distances.items():
        if k==root:
            nodes[root]=0
            
        else:
            nodes[k]=1.0

    node2distances = sorted(node2distances.items(), key=lambda x:(-x[1]))
    for k,v in sorted(node2distances):
        
        if(node2paths[k]>1):
            n=1/node2paths[k]
            for i in range(len(node2parents[k])):
                nodes[node2parents[i]] = nodes[node2parents[i]]+n
                nodes[k]=n
            '''
            for nn in node2parents[k]:
                nodes[nn]=nodes[nn]+n
                nodes[k]=n
            '''
        elif(node2paths[k]==1):
            
            for i in range(len(node2parents[k])):
                nodes[node2parents[i]]=nodes[node2parents[i]]+nodes[k]
            '''
            for nn in node2parents[k]:
                nodes[nn]= nodes[nn]+nodes[k]
            ''' 
    
    for k,v in nodes.items(): 
        
        for i in range(len(node2parents[k])):
            results.append(k)
            results.append(node2parents[i])
            results=sorted(results)
            credits[(results[0],results[1])]=nodes[k]
            results.clear()
        
        
        '''
        for nn in node2parents[k]:
            results.append(k)
            results.append(nn)
            results = sorted(results)
            credits[(results[0],results[1])]=nodes[k]
            results.clear()
        '''
    return credits



def bfs(graph, root, max_depth):
    node2num_paths = defaultdict(int)
    node2num_paths[root] = 1
    node2distances = defaultdict(int)
    node2distances[root] = 0
    node2parents = defaultdict(list)
    #print(node2parents)
    #print("this is bfs")
    
    def bfs_shortest_path(graph, start,node2num_paths,node2distances,node2parents,max_depth):
        print(start)
        explored = set()
        q=deque()
        explored.add(start)
        q.append(start)
 
    #if start == end:
      #explored.append(start)
    #return
        print("bfs_sp",q)
        while q:
            node = q.popleft()
            for neighbor in graph.neighbors(node):
                if neighbor in explored and node2distances[neighbor] == (node2distances[node]+1):
                    node2parents[neighbor].append(node)
                    node2num_paths[neighbor] += 1
                elif neighbor not in explored and node2distances[node] < max_depth:
                    q.append(neighbor)
                    explored.add(neighbor)
                    node2parents[neighbor].append(node)
                    node2num_paths[neighbor] = 1
                    node2distances[neighbor]=node2distances[node]+1
        return bfs_shortest_path(graph, root,node2num_paths,node2distances,node2parents,max_depth)
    
    return node2distances,node2num_paths,node2parents
 

    pass


def approximate_betweenness(graph, max_depth):
    '''
    
    result=[]
    dict1 = {}
    for i in range(0,max_depth):
          gaand=bfs(graph,nodes[i],max_depth)
          for j in range(len(gaand)):
                result.append(gaand[j])
    for i in range(len(result)):
          dict1[result[i][0]]=result[i][1]/2
               
    return dict1
    pass
    
    '''
    
    
    nodes = graph.nodes()
    betweeness=defaultdict(float)
    
    for i in range(len(nodes)):
        node2distances,node2num_paths,node2parents1=bfs(graph,nodes[i],max_depth)
        result=bottom_up(nodes[i], node2distances, node2num_paths, node2parents1)
        
        for k,v in result.items():
            betweeness[k[0],k[1]]=betweeness[k[0],k[1]]+(v)
            
        
    
    
    '''
    for node in nodes:
        node2distances,node2num_paths,node2parents1=bfs(graph,node,max_depth)
        result=bottom_up(node, node2distances, node2num_paths, node2parents1)
    
        for k,v in result.items():
            betweeness[k[0],k[1]]=betweeness[k[0],k[1]]+(v)
    '''
    for k,v in betweeness.items():
        betweeness[k[0],k[1]]=v/2
    return betweeness


def partition_girvan_newman(graph, max_depth):
    
    graph_copy = graph.copy()
  
    components = [c for c in nx.connected_component_subgraphs(graph_copy)]
 
    betweeness = approximate_betweenness(graph_copy, max_depth)
  
    count=0
    betweeness_1 = sorted(betweeness.items(),key=lambda tup:(-tup[1],tup[0]))
    if(max_depth!=1):
        while (len(components)==1):
            graph_copy.remove_edge(*(betweeness_1[count][0]))
            count+=1
            components = [c for c in nx.connected_component_subgraphs(graph1)]
    
        return components
  
    if(max_depth==1):
        i=0
        while(len(components)==1 and i<len(list(betweeness.keys()))):
            graph_copy.remove_edge(*list(betweeness.keys())[i])
            components = [c for c in nx.connected_component_subgraphs(graph_copy)]
            i=i+1
        return components


def cluster(graph):
    #print(graph.nodes())
    clusters = partition_girvan_newman(graph, 1)
    length_cluster = len(clusters)
    pickle.dump(length_cluster,open("cluster_length.txt","wb"))
    print(len(graph.nodes()))
    cluster_order=[]
    for i in range(len(clusters)):
        cluster_order.append(clusters[i].order())
    pickle.dump(cluster_order,open("cluster_order.txt","wb"))
    pass

def main():
    friends_dict_with_ids = pickle.load(open("friends_followers_ids.txt","rb"))
    user_names=[]
    user_ids=[]
    for k,v in friends_dict_with_ids.items():
        user_names.append(k)
        user_ids.append(v)
    '''
    counter =0
    for i in range(1,10):
        counter+=len(user_ids[i])
    print(counter)
    ''' 
    
    graph=create_graph(user_names,user_ids)
    cluster(graph)
    pass

if __name__ == "__main__":
    main()