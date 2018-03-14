import warnings
warnings.filterwarnings("ignore")
import networkx as nx
import urllib.request 
import matplotlib.pyplot as plt


def plot_subgraph(subgraph, target_nodes):
    """
    Plot this subgraph of nodes, coloring
    the specified list of target_nodes in red.
    """
    nodes = list(subgraph.nodes())
    colors = ['b'] * len(nodes)
    for n in target_nodes:
        idx = nodes.index(n)
        colors[idx] = 'r'
    sizes = [800] * len(nodes)
    sizes[idx] = 1000
    plt.figure(figsize=(10,10))
    plt.axis('off')
    nx.draw_networkx(subgraph, nodelist=nodes, with_labels=True,width=1, node_color=colors,node_size=sizes, alpha=.5)
    plt.show()


def get_subgraph(graph, nodes, n=100):
    """ Get the subgraph consisting of a list node
    and their neighbors, plus their neighbors'
    neighbors, up to $n$ total nodes"""
    neighbors = set()
    for ni in nodes:
        neighbors |= set(graph.neighbors(ni))
    # plot at least the target node and his neighbors.
    result = set(nodes) | neighbors
    # add "friends of friends" up to n total nodes.
    for x in neighbors:
        # how many more nodes can we add?
        maxsize = n - len(result) 
        toadd = set(graph.neighbors(x)) - result
        result.update(list(toadd)[:maxsize])
        if len(result) > n:
            break
    return graph.subgraph(result)



def jaccard_wt(graph, node):
    neighbors = set(graph.neighbors(node))
    scores = []
    
    for n in graph.nodes():
        numerator=[]
        denominatori=[]
        denominatorj=[]
        if n!= node and not graph.has_edge(node,n):
            neighbours2 = set(graph.neighbors(n))
            
            for val in neighbors:
                for val2 in neighbours2:
                    if val ==val2:
                        numerator.append(1.0/float(graph.degree(val)))
            num_temp = sum(numerator)
            
            for val in neighbors:
                denominatori.append(graph.degree(val))
            
            denom1 = sum(denominatori)
            denom1 =1/denom1
            
            for val in neighbours2:
                denominatorj.append(graph.degree(val))
            denom2 = sum(denominatorj)
            denom2 = 1/denom2
            
            score_temp = num_temp/(denom1+denom2)
            
            scores.append(((node,n),score_temp))
    
    return scores



def main():
    urllib.request.urlretrieve("http://snap.stanford.edu/data/ca-GrQc.txt.gz", "ca-GrQc.txt.gz")
    graph = nx.read_edgelist('ca-GrQc.txt.gz')
    nodes = list(graph.nodes())
    degrees = graph.degree()

    author = sorted(degrees,key=lambda x: x[-1],reverse=True)[1]
    subgraph = get_subgraph(graph, [author], n=30)
    plot_subgraph(subgraph, [author])

    scores = jaccard_wt(graph,author)
    print(scores[:50])




if __name__ == "__main__":
    main()