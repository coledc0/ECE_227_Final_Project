import networkx as nx
import numpy as np
import random
import os

class Player:
    """
    Creates a "Player" with a strategy of cooperative or defective, which can be used as a Graph node.
    Probability of cooperative player is specified by c_weight.
    """
    def __init__(self, id, c_weight=0.5, strategy=None, score=0):
        strategies = ['C', 'D']
        assert (0<=c_weight<=1), 'weight is not between 0 and 1'
        weights = [c_weight, 1-c_weight] #input weight is for cooperative
        self.id = id
        if strategy is None:
            self.strategy = random.choices(strategies, weights=weights)[0]
        else:
            self.strategy = strategy
        self.score = score

    def __repr__(self):
        return f"Player_{self.id}"

    def __eq__(self, other):
        return isinstance(other, Player) and self.id == other.id

    def __hash__(self):
        return hash(self.id)



def reassign_players(G, c_weight=0.5):
    """
    Takes a graph G with Player nodes and reinitializes their strategies in-place.
    Probability of cooperative player is specified by c_weight.
    """
    weights = [c_weight,1-c_weight]
    for node in G.nodes:
        strat = random.choices(['C','D'], weights=weights)[0]
        node.strategy = strat



def transform_to_player(G, c_weight=0.5, pos=None):
    """
    Takes arbitrary graph G and returns a copy with Player nodes with random strategy.
    P(player is cooperative) = c_weight; node positions can be specified with dictionary pos.
    """
    # set positions
    if pos:
        try:
            nx.set_node_attributes(G,pos,"pos")
        except:
            print("Invalid position dictionary")
            
    # Convert nodes to Player objects
    node_mapping = {}
    for node in G.nodes():
        player = Player(node,c_weight)
        node_mapping[node] = player

    # Create new graph with Player nodes
    player_graph = nx.relabel_nodes(G, node_mapping)
    
    return player_graph


'''GRAPH GENERATING FUNCTIONS'''
#standard random geometric graph
def generate_scaled_random_geometric_graph(N, R, c_weight=0.5):
    """
    Creates a random geometric graph object with N players and a connection radius of R.
    Probability of cooperative player is specified by c_weight.
    """
    side_length = np.sqrt(N)
    players = [Player(i, c_weight=c_weight) for i in range(N)]
    pos = {player : (np.random.uniform(0, side_length), np.random.uniform(0, side_length)) for player in players}
    
    G = nx.Graph()
    G.add_nodes_from(players)
    nx.set_node_attributes(G, pos, 'pos')

    #turns out there's a built-in way to do this that's way faster 
    G.add_edges_from(nx.geometric_edges(G, radius=R))
    return G



def glue_edges_geo(G,R,inplace=False,keep_shape=True):
    """
    Adds "boundary break" edges for geometric graph.
    """
    g1 = G if inplace else copy.deepcopy(G)
    dist = G.number_of_nodes()
    dist = dist**0.5
    
    for node in g1.nodes:
        pos0,pos1 = g1.nodes[node]["pos"]
        pos0 = pos0+dist if pos0<(dist/2) else pos0
        pos1 = pos1+dist if pos1<(dist/2) else pos1
        g1.nodes[node]["pos"] = (pos0,pos1)
    
    g1.add_edges_from(nx.geometric_edges(g1, radius=R))

    if keep_shape:
        for node in g1.nodes:
            pos0,pos1 = g1.nodes[node]["pos"]
            pos0 = pos0-dist if pos0>dist else pos0
            pos1 = pos1-dist if pos1>dist else pos1
            g1.nodes[node]["pos"] = (pos0,pos1) 
    return g1



#Random grid
def generate_random_grid_graph(grid_size=(10, 10), edge_prob=0.8, c_weight=0.5):
    """
    Generates a 2D grid graph with optional edge removal. Nodes are mapped to Player objects.
    """
    G = nx.grid_2d_graph(*grid_size)
    
    # Store original positions BEFORE inserting players
    original_nodes = list(G.nodes)
    pos = {node: (node[0], node[1]) for node in original_nodes}

    # Create Player objects and relabel
    players = [Player(i, c_weight=c_weight) for i in range(len(original_nodes))]
    mapping = {node: players[i] for i, node in enumerate(original_nodes)}
    G = nx.relabel_nodes(G, mapping)

    # Remap the pos dictionary to use Player objects as keys
    pos = {mapping[node]: coords for node, coords in pos.items()}
    nx.set_node_attributes(G, pos, 'pos')

    # Randomly drop edges
    for u, v in list(G.edges):
        if np.random.rand() > edge_prob:
            G.remove_edge(u, v)

    return G


#random grid with long range connections
def generate_random_grid_with_long_range(grid_size=(10, 10), edge_prob=0.8, long_range_fraction=0.05, c_weight=0.5):
    """
    Generates a grid-based graph with random missing edges and added long-range connections.
    """
    G = generate_random_grid_graph(grid_size, edge_prob, c_weight)
    nodes = list(G.nodes)
    num_long_edges = int(long_range_fraction * len(nodes))
    added = 0
    while added < num_long_edges:
        u, v = np.random.choice(nodes, 2, replace=False)
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            added += 1
    return G


#random graph
def generate_random(n,p,c_weight=0.5):
    G = nx.fast_gnp_random_graph(n,p)
    return transform_to_player(G,c_weight)


#scale-free
def generate_scalefree(n,alpha=0.4,c_weight=0.5):
    G = nx.scale_free_graph(n,alpha=alpha,beta=1-2*alpha,gamma=alpha)
    G = G.to_undirected()
    return transform_to_player(G,c_weight)

###
# DATALOADER STUFF
###

def load_twitter_graph(file_path, central_node_id, c_weight=0.5):
    '''
        loads a single egonet from the twitter ego graph. 

        Args:
            file_path : the file path of the downloaded twitter graph. 
            central_node_id : the ID of the egonet that is being loaded. Choose any ID that has a nodeID.edges file in the folder. 
            c_weight : probability a player is initialized with cooperative game strategy.
    '''
    G = nx.Graph()
    id_to_player = {}

    with open(file_path, 'r') as f:
        for line in f:
            a_id, b_id = line.strip().split()

            # Create Player objects if they don't already exist
            if a_id not in id_to_player:
                id_to_player[a_id] = Player(a_id, c_weight)
            if b_id not in id_to_player:
                id_to_player[b_id] = Player(b_id, c_weight)

            # Just add the edge directly — duplicates are handled by networkx
            G.add_edge(id_to_player[a_id], id_to_player[b_id])

    # Add central node
    central_player = Player(central_node_id, c_weight)
    id_to_player[central_node_id] = central_player
    G.add_node(central_player)

    # Connect central node to all others
    for player in G.nodes():
        if player != central_player:
            G.add_edge(central_player, player)
    print(f'number of nodes: {len(id_to_player)}')
    return G

import os
import networkx as nx

def load_full_twitter_graph_from_folder(folder_path, c_weight=0.5):
    '''
        Loads the full twitter graph from the downloaded data. Drawing this graph using the draw_graph function takes a VERY long time, so it 
        is strongly suggested to use an external program instead.
    '''
    G = nx.Graph()
    id_to_player = {}

    # List all files ending with ".edges" in the folder
    for filename in os.listdir(folder_path):
        if not filename.endswith(".edges"):
            continue

        # Extract central_node_id from filename (everything before ".edges")
        central_node_id = filename.split(".edges")[0]

        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as f:
            for line in f:
                a_id, b_id = line.strip().split()

                # Create Player objects if they don't exist
                if a_id not in id_to_player:
                    id_to_player[a_id] = Player(a_id, c_weight)
                if b_id not in id_to_player:
                    id_to_player[b_id] = Player(b_id, c_weight)

                G.add_edge(id_to_player[a_id], id_to_player[b_id])

        # Add central node if missing
        if central_node_id not in id_to_player:
            id_to_player[central_node_id] = Player(central_node_id, c_weight)
        central_player = id_to_player[central_node_id]

        # Add central node to graph and connect to all nodes in this file
        G.add_node(central_player)

        # Connect central node to every other node in this file's edges (including a_id and b_id)
        # To avoid connecting central_player to nodes from other files, collect the nodes for this file:
        nodes_in_file = set()
        with open(file_path, 'r') as f:
            for line in f:
                a_id, b_id = line.strip().split()
                nodes_in_file.add(id_to_player[a_id])
                nodes_in_file.add(id_to_player[b_id])

        for node in nodes_in_file:
            if node != central_player:
                G.add_edge(central_player, node)

    return G

import networkx as nx

def load_twitter_graph_from_files(file_list, c_weight=0.5):
    '''
        Loads a number of egonets specified in file_list. Primary purpose is to allow construction of subgraphs of the full egonet twitter graph.

        Args:
            file_list : a list of file names in the form of 'nodeID.edges', where the nodeID corresponds to the ego node connected to all other
            nodes in the subnet.
            c_weight : probability a player is initialized with cooperative game strategy.
    '''
    G = nx.Graph()
    id_to_player = {}

    for file_path in file_list:
        # Extract central_node_id from filename (assuming filename ends with .edges)
        central_node_id = file_path.split("/")[-1].split(".edges")[0]

        with open(file_path, 'r') as f:
            nodes_in_file = set()
            for line in f:
                a_id, b_id = line.strip().split()

                # Create Player objects if missing
                if a_id not in id_to_player:
                    id_to_player[a_id] = Player(a_id, c_weight)
                if b_id not in id_to_player:
                    id_to_player[b_id] = Player(b_id, c_weight)

                a_player = id_to_player[a_id]
                b_player = id_to_player[b_id]
                G.add_edge(a_player, b_player)

                nodes_in_file.add(a_player)
                nodes_in_file.add(b_player)

        # Add central node if missing
        if central_node_id not in id_to_player:
            id_to_player[central_node_id] = Player(central_node_id, c_weight)
        central_player = id_to_player[central_node_id]
        G.add_node(central_player)

        # Connect central node to all nodes in the current file
        for node in nodes_in_file:
            if node != central_player:
                G.add_edge(central_player, node)

    return G


def load_facebook_data_from_file(filepath):
    """
    Load Facebook social circles data from a local file.

    Args:
        filepath: Path to the edge list file (e.g., "facebook_combined.txt" or "0.edges")

    Returns:
        NetworkX graph with Player nodes
    """
    try:
        print(f"Loading Facebook network from {filepath}...")
        G = nx.read_edgelist(filepath, create_using=nx.Graph(), nodetype=int)
        print(f"Successfully loaded Facebook network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    except FileNotFoundError:
        print(f"File {filepath} not found, creating sample network...")
        G = create_sample_social_network()
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        print("Creating sample network...")
        G = create_sample_social_network()
    return transform_to_player(G)



def create_sample_social_network(n=100):
    """
    Create a sample social network if Facebook data can't be downloaded.
    Uses a combination of preferential attachment and small-world properties.
    """
    print("Creating sample social network...")

    # Start with a small clique
    G = nx.complete_graph(5)

    # Add nodes with preferential attachment
    for i in range(5, n):
        # Choose nodes to connect to based on degree (preferential attachment)
        degrees = dict(G.degree())
        total_degree = sum(degrees.values())

        if total_degree == 0:
            # Connect to a random node if no edges exist
            target = random.choice(list(G.nodes()))
            G.add_edge(i, target)
        else:
            # Preferential attachment: higher degree nodes more likely to be chosen
            num_connections = random.randint(1, min(3, len(G.nodes())))

            for _ in range(num_connections):
                # Weighted random selection based on degree
                print("keys:" + str(degrees.keys()))
                probabilities = [degrees[node] / total_degree for node in G.nodes()]
                target = np.random.choice(list(G.nodes()), p=probabilities)
                G.add_edge(i, target)
                degrees[target] += 1
                total_degree += 2

    # Add some random edges to increase clustering
    num_random_edges = n // 10
    for _ in range(num_random_edges):
        u, v = random.sample(list(G.nodes()), 2)
        G.add_edge(u, v)

    return G