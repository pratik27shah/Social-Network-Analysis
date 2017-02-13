# coding: utf-8
from collections import  defaultdict, deque
import networkx as nx
import matplotlib.pyplot as plt
import urllib.request
from _overlapped import INFINITE
from facepy import GraphAPI
import sys
import configparser



def girvan_newman(G, maxsize=100):

    
        # eb is dict of (edge, score) pairs, where higher is better
    def find_best_edge(G0):  
        eb = nx.edge_betweenness_centrality(G)
        return sorted(eb.items(), key=lambda x: x[1], reverse=True)
    components = [c for c in nx.connected_component_subgraphs(G)]
   
    scores=find_best_edge(G)
    edges_to_removed = 0
    final_result= []
    
    for index in range(len(scores)):
            edge_to_remove = scores[index][0]
            G.remove_edge(*edge_to_remove)
            edges_to_removed+= 1
            components = [c for c in nx.connected_component_subgraphs(G)]
            if len(components) != 1:
                break
    
   # print 'removed',edges_to_removed, 'edges'
    result =[c.nodes() for c in components]
    #print 'component sizes=', [len(result[0]),len(result[1])]
    
    for c in components:
        nodes_check = c.nodes()
        if (len(nodes_check)<= maxsize):
            final_result.append(nodes_check)
          #  print 'stopping for', len(nodes_check)
        elif (len(nodes_check)> maxsize):
            final_result.extend(girvan_newman(c))
                            
    

    return final_result


def get_subgraph(graph, min_degree):
    """Return a subgraph containing nodes whose degree is
    greater than or equal to min_degree.
    We'll use this in the main method to prune the original graph.

    Params:
      graph........a networkx graph
      min_degree...degree threshold
    Returns:
      a networkx graph, filtered as defined above.

    >>> subgraph = get_subgraph(example_graph(), 3)
    >>> sorted(subgraph.nodes())
    ['B', 'D', 'F']
    >>> len(subgraph.edges())
    2
    """
    degree = graph.degree()
    nodes = degree.items()
    subgraph = sorted(nodes, key = lambda nodes:[1], reverse=True)
    for sbg in subgraph:
        if sbg[1]>=min_degree:
            print
        else:
            graph.remove_node(sbg[0])
    ###TODO
    return graph   
    pass


## Link prediction

# Next, we'll consider the link prediction problem. In particular,
# we will remove 5 of the accounts that Bill Gates likes and
# compute our accuracy at recovering those links.

def make_training_graph(graph, test_node, n):
    """
    To make a training graph, we need to remove n edges from the graph.
    As in lecture, we'll assume there is a test_node for which we will
    remove some edges. Remove the edges to the first n neighbors of
    test_node, where the neighbors are sorted alphabetically.
    E.g., if 'A' has neighbors 'B' and 'C', and n=1, then the edge
    ('A', 'B') will be removed.

    Be sure to *copy* the input graph prior to removing edges.

    Params:
      graph.......a networkx Graph
      test_node...a string representing one node in the graph whose
                  edges will be removed.
      n...........the number of edges to remove.

    Returns:
      A *new* networkx Graph with n edges removed.

    In this doctest, we remove edges for two friends of D:
    >>> g = example_graph()
    >>> sorted(g.neighbors('D'))
    ['B', 'E', 'F', 'G']
    >>> train_graph = make_training_graph(g, 'D', 2)
    >>> sorted(train_graph.neighbors('D'))
    ['F', 'G']
    """
    train_graph=graph.copy()
    list=train_graph.neighbors(test_node)
    list.sort()
    i=0
    for value in list:
        if i>=n:
            break
        else:
            train_graph.remove_edge(test_node,value)
        i=i+1
    return train_graph
    ###TODO
    pass



def jaccard(graph, node, k):
    """
    Compute the k highest scoring edges to add to this node based on
    the Jaccard similarity measure.
    Note that we don't return scores for edges that already appear in the graph.

    Params:
      graph....a networkx graph
      node.....a node in the graph (a string) to recommend links for.
      k........the number of links to recommend.

    Returns:
      A list of tuples in descending order of score representing the
      recommended new edges. Ties are broken by
      alphabetical order of the terminal node in the edge.

    In this example below, we remove edges (D, B) and (D, E) from the
    example graph. The top two edges to add according to Jaccard are
    (D, E), with score 0.5, and (D, A), with score 0. (Note that all the
    other remaining edges have score 0, but 'A' is first alphabetically.)

    >>> g = example_graph()
    >>> train_graph = make_training_graph(g, 'D', 2)
    >>> jaccard(train_graph, 'D', 2)
    [(('D', 'E'), 0.5), (('D', 'A'), 0.0)]
    """
    edge = []
    neighborsa = set(graph.neighbors(node))
    #neighborsa = set(neighborsa)
    for nodes in graph.nodes():
        if nodes != node and not graph.has_edge(node, nodes):
            neighborsb = set(graph.neighbors(nodes))
            score = float(len(neighborsa & neighborsb)) / (len(neighborsa | neighborsb))
            edge.append(((node, nodes), score))
    value = sorted(edge, key = lambda a:(-a[1],a[0][1]))[:k]
    return value        
    ###TODO
    pass


# One limitation of Jaccard is that it only has non-zero values for nodes two hops away.
#
# Implement a new link prediction function that computes the similarity between two nodes $x$ and $y$  as follows:
#
# $$
# s(x,y) = \beta^i n_{x,y,i}
# $$
#
# where
# - $\beta \in [0,1]$ is a user-provided parameter
# - $i$ is the length of the shortest path from $x$ to $y$
# - $n_{x,y,i}$ is the number of shortest paths between $x$ and $y$ with length $i$




def evaluate(scores, graph, n=10):
    predicted_edges = [x[0] for x in sorted(scores, key=lambda x: x[1], reverse=True)[:n]]
    return 1. * len([x for x in predicted_edges if graph.has_edge(*x)]) / len(predicted_edges)    ###TODO
    pass


import configparser
from facepy import GraphAPI
def read_graph():
    """ Read 'edges.txt.gz' into a networkx **undirected** graph.
    Done for you.
    Returns:
      A networkx undirected graph.
    """
    G = nx.read_edgelist("edges.txt",delimiter = "\t",create_using = nx.Graph())
    
    return G
    
def draw_network(graph, draw_thresh=1, label_thresh=27,
                 min_node_sz=30, max_node_sz=200,savefig="graphs.png",length=-1):
    if(length==-1 or length>10):
        if(length>10 and length<=26):
            label_thresh=5
        degrees = graph.degree()
        labels = {n: n for n, d in degrees.items() if d > label_thresh}
        plt.figure(figsize=(10,10))
        nodes_to_draw = [name for name, degree in degrees.items() if degree > draw_thresh]
        maxdegree = max(degrees.values())
        sz_range = max_node_sz - min_node_sz
        sizes = [min_node_sz + (1. * degrees[n] / maxdegree * sz_range)
                 for n in nodes_to_draw]
        subgraph = graph.subgraph(nodes_to_draw)
        nx.draw_networkx(subgraph, alpha=.3, width=.3,
                         labels=labels,with_labels = True, node_size=sizes)
        plt.axis("off")
      #  plt.show()
        plt.savefig(savefig)
    

import io
def main():
    """
    FYI: This takes ~10-15 seconds to run on my laptop.
    """

    utf=io.open('cluster.txt', 'w', encoding='utf8')
    utf.close()
    utf=io.open('cluster.txt', 'a', encoding='utf8')
    #utf=io.open('summary.txt', 'a', encoding='utf8')
    config = configparser.ConfigParser()
    config.read("configure.txt")
    graph = GraphAPI(config.get("facebook","USER_TOKEN"))
    FbpageDetails = graph.get(config.get("facebook","Trump"), page=False, retry=3)
    
    PageName=FbpageDetails["name"]
    graph = read_graph()
    draw_network(graph)
    print('graph has %d nodes and %d edges' %
          (graph.order(), graph.number_of_edges()))
    utf.write("\nTotal Graph nodes and edges="+str(graph.order())+" "+str( graph.number_of_edges()))
    subgraph = get_subgraph(graph, 2)
    print('subgraph has %d nodes and %d edges' %
          (subgraph.order(), subgraph.number_of_edges()))
    utf.write("\nTotal Graph nodes and edges="+str(subgraph.order())+" "+str( subgraph.number_of_edges()))
    result = girvan_newman(subgraph.copy())# minsize=10, maxsize=100)
    i=0
    print("Graphs of Clusters greater then size 10 only shown");
    clusterlabel=[]
    for cluster in result:
        i=i+1
        figs=str(i)+".png"
        degrees = (graph.subgraph(cluster)).degree()
        labels = {n: n for n, d in sorted(degrees.items()) }
        if(len(labels)>1):
            clusterlabel.append(len(labels))
      
        draw_network(graph.subgraph(cluster), label_thresh=5, draw_thresh=0,savefig=figs,length=len(labels))
    print("Cluster Created of Sizes, -->",str(clusterlabel))  
    utf.write("\n Cluster Created of Sizes="+str(clusterlabel))
    utf.write("\n Number of communities discovered="+str(len(clusterlabel)))
    utf.write("\n Average number of users per community="+str((subgraph.order()*1.0)/len(clusterlabel)))
    print("\n Average number of users per community="+str((subgraph.order()*1.0)/len(clusterlabel))) 
  #  utf.write("Cluster Created of Sizes, -->"+clusterlabel) 
    test_node = PageName
    train_graph = make_training_graph(subgraph, test_node, 5)
    jaccard_scores = jaccard(train_graph, test_node,5)
    print('jaccard accuracy=%.3f' % evaluate(jaccard_scores, subgraph, n=10))
    utf.write('\n jaccard accuracy=%.3f' % evaluate(jaccard_scores, subgraph, n=10))
    utf.close()
   # print('top 5 recommendations:')
   # values=[x[0][1] for x in sorted(jaccard_scores, key=lambda x: x[1], reverse=True)[:5]] 
   # print(values)
if __name__ == '__main__':
    main()
