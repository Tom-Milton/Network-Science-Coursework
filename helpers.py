import gzip
import copy
import random
import networkx as nx
from tqdm import tqdm
from scipy.stats import pearsonr
from string import ascii_lowercase as lowercase

def generate_graph(words):
    G = nx.Graph(name="words")
    lookup = {c: lowercase.index(c) for c in lowercase}

    def edit_distance_one(word):
        for i in range(len(word)):
            left, c, right = word[0:i], word[i], word[i + 1 :]
            j = lookup[c]  # lowercase.index(c)
            for cc in lowercase[j + 1 :]:
                yield left + cc + right

    candgen = (
        (word, cand)
        for word in sorted(words)
        for cand in edit_distance_one(word)
        if cand in words
    )
    G.add_nodes_from(words)
    for word, cand in candgen:
        G.add_edge(word, cand)
    return G

def words_graph(graph_gz):
    """Return the words example graph from the Stanford GraphBase"""
    fh = gzip.open(graph_gz)
    words = set()
    for line in fh.readlines():
        line = line.decode()
        if line.startswith("*"):
            continue
        w = str(line[0:5])
        words.add(w)
    return generate_graph(words)

def make_ws_graph(num_nodes, clockwise_neighbours, rewiring_prob):
    """Returns a dictionary to a undirected graph with num_nodes nodes; keys are nodes, values are list of neighbours.
    The nodes of the graph are numbered 0 to num_nodes - 1.
    Node i initially joined to i+1, i+2, ... , i+d mod N and i-1, i-2, ... , i-d mod N
    where d is the no. of clockwise neighbours.
    Each edge from i to j replaced with probability given with edge from i to randomly chosen k
    """
    
    #initialize empty graph
    ws_graph = {}
    for vertex in range(num_nodes): ws_graph[vertex] = []

    #add edges from each vertex to clockwise neighbours
    for vertex in range(num_nodes):                                             #consider each vertex
        for neighbour in range(vertex + 1, vertex + clockwise_neighbours + 1):  #consider each clockwise neighbour
            neighbour = neighbour % num_nodes                                   #correct node label if value too high
            ws_graph[vertex] += [neighbour]                                     #add edge to dictionary
            ws_graph[neighbour] += [vertex]                                     #and again (each edge corresponds to two adjancencies)

    #rewiring
    for vertex in range(num_nodes):                                             #consider each vertex
        for neighbour in range(vertex + 1, vertex + clockwise_neighbours + 1):  #consider each clockwise neighbour
            neighbour = neighbour % num_nodes                                   #correct node label if value too high
            random_number = random.random()                                     #generate random number
            if random_number < rewiring_prob:                                   #decide whether to rewire
                random_node = random.randint(0, num_nodes-1)                    #choose random node
                if random_node != vertex and random_node not in ws_graph[vertex]:   #make sure no loops or duplicate edges
                    ws_graph[vertex].remove(neighbour)                          #delete edge from dictionary          
                    ws_graph[neighbour].remove(vertex)                          #in two places
                    ws_graph[vertex] += [random_node]                           #add new edge to dictionary
                    ws_graph[random_node] += [vertex]                           #in two places
    
    return ws_graph
    
class PATrial:
    """
    Used when each new node is added in creation of a PA graph.
    Maintains a list of node numbers with multiple instances of each number.
    The number of instances of each node number are in proportion to the
    probability that it is linked to.
    Uses random.choice() to select a node number from this list for each trial.
    """

    def __init__(self, num_nodes):
        """
        Initialize a PATrial object corresponding to a 
        complete graph with num_nodes nodes
        
        Note the initial list of node numbers has num_nodes copies of
        each node number
        """
        self._num_nodes = num_nodes #note that the vertices are labelled from 0 so self._num_nodes is the label of the next vertex to be added
        self._node_numbers = [node for node in range(num_nodes) for dummy_idx in range(num_nodes)]


    def run_trial(self, num_nodes):
        """
        Conduct num_node trials using by applying random.choice()
        to the list of node numbers
        
        Updates the list of node numbers so that the number of instances of
        each node number is in the same ratio as the desired probabilities
        
        Returns:
        Set of nodes
        """       
        #compute the neighbors for the newly-created node
        new_node_neighbors = set()
        for dummy_idx in range(num_nodes):
            new_node_neighbors.add(random.choice(self._node_numbers))
        # update the list of node numbers so that each node number 
        # appears in the correct ratio
        self._node_numbers.extend(list(new_node_neighbors))
        # also add to the list of node numbers the id of the current node
        # since each node must appear once in the list else no future node will link to it
        # note that self._node_numbers will next be incremented
        self._node_numbers.append(self._num_nodes)
        # update the number of nodes
        self._num_nodes += 1
        return new_node_neighbors
    
def make_complete_graph(num_nodes):
    """Takes the number of nodes num_nodes and returns a dictionary
    corresponding to a complete directed graph with the specified number of
    nodes. A complete graph contains all possible edges subject to the
    restriction that self-loops are not allowed. The nodes of the graph should
    be numbered 0 to num_nodes - 1 when num_nodes is positive. Otherwise, the
    function returns a dictionary corresponding to the empty graph."""
    #initialize empty graph
    complete_graph = {}
    #consider each vertex
    for vertex in range(num_nodes):
        #add vertex with list of neighbours
        complete_graph[vertex] = set([j for j in range(num_nodes) if j != vertex])
    return complete_graph
    
def make_PA_Graph(total_nodes, out_degree):
    """creates a PA_Graph on total_nodes where each vertex is iteratively
    connected to a number of existing nodes equal to out_degree"""
    #initialize graph by creating complete graph and trial object
    PA_graph = make_complete_graph(out_degree)
    trial = PATrial(out_degree)
    for vertex in range(out_degree, total_nodes):
        PA_graph[vertex] = trial.run_trial(out_degree)
    return PA_graph

def load_graph(graph_txt):
    """
    Loads a graph from a text file.
    Then returns the graph as a dictionary.
    """
    graph = open(graph_txt)
    
    answer_graph = {}
    nodes = 0
    for line in graph:
        neighbors = line.split(' ')
        node = int(neighbors[0])
        answer_graph[node] = set([])
        for neighbor in neighbors[1 : -1]:
            answer_graph[node].add(int(neighbor))
        nodes += 1
    
    return answer_graph

def compute_in_degrees(digraph):
    """Takes a directed graph and computes the in-degrees for the nodes in the
    graph. Returns a dictionary with the same set of keys (nodes) and the
    values are the in-degrees."""
    #initialize in-degrees dictionary with zero values for all vertices
    in_degree = {}
        
    for vertex in digraph:
        in_degree[vertex] = 0
    #consider each vertex
    for vertex in digraph:
        #amend in_degree[w] for each outgoing edge from v to w
        for neighbour in digraph[vertex]:
            in_degree[neighbour] += 1
    return in_degree

def compute_degrees(digraph):
    """Takes a directed graph and computes the (in+out) degrees for the nodes in the graph. 
    Returns a dictionary with the same set of keys (nodes) and their (in+out) degrees."""

    # Initialize degree dictionary with all out-degrees
    degrees = {k: len(v) for k, v in digraph.items()}
    # Add in-degrees for each incoming edge
    for v in digraph:
        for w in digraph[v]:
            degrees[w] += 1
    return degrees

def compute_neighbourhoods(digraph):
    """Takes a directed graph and computes the neighbourhood for the nodes in the graph. 
    Returns a dictionary with the same set of keys (nodes) and their neighbourhoods."""
    
    # Initialize neighbourhood dictionary with all out-neighbours
    neighbourhoods = copy.deepcopy(digraph)
    # Add in-neighbours of each incoming edge
    for v in digraph:
        for w in digraph[v]:
            neighbourhoods[w].add(v)
    return neighbourhoods

def compute_local_clustering_coefficients(digraph, neighbourhoods, rounding=True):
    """Takes a directed graph and computes the clustering coefficient for the nodes in the graph. 
    Returns a dictionary with the same set of keys (nodes) and their clustering coefficient."""
    clustering_coefficients = {k: 0 for k in digraph.keys()}

    for v in tqdm(digraph):

        numerator = 0
        for j in neighbourhoods[v]:
            for k in digraph[j]:
                if k in neighbourhoods[v]:
                    numerator += 1

        denominator = len(neighbourhoods[v])*(len(neighbourhoods[v])-1)

        if denominator == 0: clustering_coefficients[v] = 0  # Divide by 0 error when v has 0 or 1 neighbours
        elif numerator/denominator > 1: clustering_coefficients[v] = 1  # Self referential citations can cause numerator > denominator
        elif rounding is True: clustering_coefficients[v] = round(numerator/denominator, 2)
        else: clustering_coefficients[v] = numerator/denominator

    return clustering_coefficients

def get_in_degrees_logs(total_nodes, out_degree, selected_nodes):
    """Records the in-degrees w.r.t the number of nodes added
    for each selected node. Returned as a list of lists"""
    PA_graph = make_complete_graph(out_degree)
    trial = PATrial(out_degree)

    in_degrees_logs = [[] for _ in range(len(selected_nodes))]
    for vertex in tqdm(range(out_degree, total_nodes)):
        PA_graph[vertex] = trial.run_trial(out_degree)
        in_degrees = compute_in_degrees(PA_graph)

        for selected_node, log in zip(selected_nodes, in_degrees_logs):
            if vertex >= selected_node:  # Selected node has already been added
                log.append(in_degrees[selected_node])
            else:  # Selected node not yet been added to graph
                log.append(0)
        
    return in_degrees_logs

def make_C_Graph(n, m, p1, p2, p3, p4, selected_nodes=[]):
    """Construct a C graph as defined in the question"""
    C_graph = make_complete_graph(m)
    degrees_logs = [[] for _ in range(len(selected_nodes))]

    for vertex in tqdm(range(n-m)):
        degrees = compute_degrees(C_graph)

        # Add new node v joined to m existing nodes with probability proportional to their degree   
        if random.random() < p1:
            neighbours = random.choices(list(degrees.keys()), weights=degrees.values(), k=m)
            C_graph[vertex] = set(neighbours)

        # Add m random edges between existing nodes with endpoints proportional to their degree
        if random.random() < p2:
            new_edges_counter = 0
            total_counter = 0
            while new_edges_counter != m:  # If we naively choose edges we may choose some that already exist
                start_node, end_node = random.choices(list(degrees.keys()), weights=degrees.values(), k=2)
                total_counter += 1
                if end_node not in C_graph[start_node]: 
                    C_graph[start_node].add(end_node)
                    new_edges_counter += 1
                if total_counter > 10_000: # Prevents halting if no more edges left to add (clique)
                    break

        # Delete an existing node chosen uniformly at random
        if random.random() < p3:
            if not len(C_graph.keys()) == 2:  # Prevents removing the last two vertices
                del_node = random.choices(list(C_graph.keys()), weights=None, k=1)[0]
                del C_graph[del_node]  # Remove node and out-edges
                for v in C_graph.values(): v.discard(del_node)  # Remove in-edges

        # Delete m existing edges chosen uniformly at random
        if random.random() < p4:
            deleted_edges_counter = 0
            total_counter = 0
            while deleted_edges_counter != m:  # If we naively choose edges we may choose some that don't exist
                start_node, end_node = random.choices(list(C_graph.keys()), weights=None, k=2)
                total_counter += 1
                if end_node in C_graph[start_node]: 
                    C_graph[start_node].discard(end_node)
                    deleted_edges_counter += 1
                if total_counter > 10_000: # Prevents halting if no more edges to remove
                    break
        
        # Preferential attachment analysis
        degrees = compute_degrees(C_graph)
        if selected_nodes is not None:
            for selected_node, log in zip(selected_nodes, degrees_logs):
                if selected_node in C_graph.keys():  # Selected node has already been added
                    log.append(degrees[selected_node])
                else:  # Selected node not yet been added to graph
                    log.append(0)

    return C_graph, degrees_logs

def compute_proxy_betweenness_centrality(digraph):
    """Takes a directed graph and computes the proxy betweenness centrality for the nodes in the graph. 
    Returns a dictionary with the same set of keys (nodes) and their proxy betweenness centrality."""
    degrees = compute_degrees(digraph)
    neighbourhoods = compute_neighbourhoods(digraph)
    clustering_coefficients = compute_local_clustering_coefficients(digraph, neighbourhoods)
    centrality = {v: degrees[v]*(1-clustering_coefficients[v]) for v in digraph.keys()}
    return centrality

def compute_betweenness_centrality(digraph):
    """Takes a directed graph and computes the betweenness centrality for the nodes in the graph. 
    Returns a dictionary with the same set of keys (nodes) and their betweenness centrality."""
    g = nx.Graph(digraph)
    centrality = nx.betweenness_centrality(g, k=100)
    return centrality

def create_custom_network(N_0, m, n, selected_nodes=[]):
    """Creates an undirected network as defined in the question"""
    degrees_logs = [[] for _ in range(len(selected_nodes))]
    
    for i in range(len(N_0), n):
        existing_node = random.choices(list(N_0.keys()), k=1)[0]
        existing_node_neighbours = random.choices(list(N_0[existing_node]), k=m)
        N_0[i] = set(existing_node_neighbours)  # Add out-edges from new node
        for v in existing_node_neighbours: N_0[v].add(i)  # Add in-edges to new node

        # Preferential attachment analysis
        degrees = compute_degrees(N_0)
        if selected_nodes is not None:
            for selected_node, log in zip(selected_nodes, degrees_logs):
                if selected_node in N_0.keys():  # Selected node has already been added
                    log.append(degrees[selected_node])
                else:  # Selected node not yet been added to graph
                    log.append(0)

    return N_0, degrees_logs

def compute_degree_centrality(undigraph):
    """Takes an undirected graph and computes the degree centrality for the nodes in the graph. 
    Returns a dictionary with the same set of keys (nodes) and their degree centrality."""
    return compute_degrees(undigraph)

def compute_closeness_centrality(undigraph):
    """Takes an undirected graph and computes the closeness centrality for the nodes in the graph. 
    Returns a dictionary with the same set of keys (nodes) and their closeness centrality."""
    g = nx.Graph(undigraph)
    return nx.closeness_centrality(g)

def compute_relatedness_index(undigraph, centrality):
    """Takes an undirected graph and computes the relatedness index for the nodes in the graph. 
    Returns a dictionary with the same set of keys (nodes) and their relatedness index."""
    g = nx.Graph(undigraph)
    x, y = [], []

    for u in undigraph:
        for v in undigraph[u]:
            x.append(centrality[u])
            y.append(centrality[v])

    return pearsonr(x, y)