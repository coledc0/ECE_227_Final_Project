"""
TODO: possibly explore different update functions?
"""

import networkx as nx
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import os
import copy

def update_score_matrix(G, score_dic, coop_bonus=0, output_avg = False):
    """
    Scores all players in the graph using the score_dictionary defined outside the function. 
    
    Args:
        G: NetworkX graph with Player nodes
        score_dic: dictionary representing score matrix with elements corresponding in order to R,S,T,P values.
                   First letter of key corresponds to player, second letter corresponds to opponent. 
                   {"DD" : P, "DC" : T, "CD" : S, "CC" : R}
        output_avg: if True, will return the average node score per behavior as a dict {'C': score, 'D': score}

    Output:
        optionally returns the average node score for each behavior class
    """
    avg_score = {'C' : 0, 'D' : 0}
    score_norm = {'C' : 0, 'D' : 0}
    for player in G.nodes:
        self_strat = player.strategy
        score_norm[self_strat] += 1
        num_friends = 0
        for neighbor in G[player]:
            score_input = self_strat + neighbor.strategy
            num_friends += (score_input=='CC')
            player.score += score_dic[score_input]
        player.score += coop_bonus*(num_friends**2)
        avg_score[self_strat] += player.score

    if output_avg:
        avg_score['C'] /= max(score_norm['C'], 1)
        avg_score['D'] /= max(score_norm['D'], 1)
        return avg_score

def update_strategies_probabilistic(G, k = 10):
    '''
    update node behavior probabilistically based on the fermi update rule described in the following paper:
    https://www.sciencedirect.com/science/article/pii/S0096300319304369

    Args:
        G: networkX graph with Player nodes having strategy = {'C', 'D'} and score attributes. 
        k: thermodynamic temperature parameter, affects the slope of the sigmoid curve. Default value of 10 empirically derived for assumed
           score matrix.
    '''
    to_switch = {}  # keep track to avoid mid-loop updates
    switch_occurred = False
    for player in G.nodes:
        neighbors = list(G.neighbors(player))
        player_strategy = player.strategy
        opposing_strategy = 'C' if player_strategy == 'D' else 'D'
        if not neighbors:
            continue  # no neighbors to learn from

        aligned_neighbors = [node for node in neighbors if node.strategy == player_strategy]
        opposed_neighbors = [node for node in neighbors if node.strategy == opposing_strategy]

        aligned_score = sum(node.score for node in aligned_neighbors) / max(len(aligned_neighbors), 1)
        if not opposed_neighbors:
            continue # no opposed neighbors to learn from
            
        best_opponent = max(opposed_neighbors, key = lambda node: node.score)
        best_opponents_aligned_neighbors = [node for node in list(G.neighbors(best_opponent)) if node.strategy == opposing_strategy]

        #if not best_opponents_aligned_neighbors:
        #    opposed_score = 0
        #else:
        opposed_score = sum(node.score for node in best_opponents_aligned_neighbors) / max(len(best_opponents_aligned_neighbors), 1)

        switch_prob = (1 + np.exp(k*(aligned_score - opposed_score)))**(-1)
        if random.random() <= switch_prob:
            to_switch[player] = opposing_strategy
            switch_occurred = True
    for player, new_strategy in to_switch.items():
        player.strategy = new_strategy
    return switch_occurred


def draw_graph(G):
    """
    Plots graph G with green cooperators & red defectors.
    Very very slow for large N if position is not specified in G.
    """
    position_active = not not nx.get_node_attributes(G, 'pos') #checks whether or not the graph is geometric.
    
    if position_active:
        pos = nx.get_node_attributes(G, 'pos')
    N = G.number_of_nodes()
    plt.figure(figsize=(6,6))
    
    strategy_colors = {'C': 'green',
                       'D': 'red'}
    node_colors = [strategy_colors.get(node.strategy, 'gray') for node in G.nodes()]

    if position_active:
        nx.draw(G, pos, node_size=max(20000//N, 1), with_labels=False, node_color=node_colors)
    else:
        nx.draw(G, node_size=max(20000//N, 1), with_labels=False, node_color=node_colors)
    # # for geometric graphs only; I deleted the N and R params for more reusability
    # plt.title(f"Random Geometric Graph (N={N}, R={R}) on √N x √N Grid")    
    plt.title(f"{N} nodes, {G.number_of_edges()} edges")
    plt.show()

def run_simulation(G, rounds, score_dic, coop_bonus=0, draw=True, output_dir=None, i=999999):
    """
    Runs update_strategies_probabilistic on graph G for specified # of rounds.

    Args:
        G: NetworkX Graph with Player nodes
        rounds: # of rounds for simulation
        score_dic: specified score matrix
        draw: Boolean for whether to draw graphs
        output_dir: directory to save graph to, if specified
    """
    if output_dir:
        print("Saving to " + output_dir)
        os.makedirs(output_dir, exist_ok=True)
        #TODO: implement this
    
    G.remove_nodes_from(list(nx.isolates(G))) #delete all nodes with no edges
    history = [False for round in range(rounds)]
    #expected_degree = sum(dict(G.degree()).values()) / N

    #get initial state
    strategies = [node.strategy for node in G.nodes]
    if draw:
        print("Initial Graph")
        print(f"Cooperators: {strategies.count('C')}, Defectors: {strategies.count('D')}")
        draw_graph(G)

    avg_score_history = []
    C_pop_hist = []
    D_pop_hist = []
    for round_num in range(1, rounds+1):
        avg_score_history.append(update_score_matrix(G, score_dic, coop_bonus=0, output_avg = True))
        switches_occurred = update_strategies_probabilistic(G, 0.5)
        history[round_num-1] = switches_occurred
        strategies = [node.strategy for node in G.nodes]
        C_pop_hist.append(strategies.count('C'))
        D_pop_hist.append(strategies.count('D'))
        #stop simulation early if convergence reached (3 rounds of no change)
        if round_num > 5:
            if len(set(history[round_num-3:round_num])) == 1 and history[round_num-1] is False:
                if draw:
                    print(f"Converged at round {round_num}; strategies:")
                    print(f"Cooperators: {strategies.count('C')}, Defectors: {strategies.count('D')}, graph changed: {switches_occurred}")
                    draw_graph(G)
                break

        #draw every i iterations
        if round_num % i == 0 and draw:
            print(f"Round {round_num} strategies:")
            print(f"Cooperators: {strategies.count('C')}, Defectors: {strategies.count('D')}, graph changed: {switches_occurred}")
            draw_graph(G)
            
        # Reset scores
        for node in G.nodes:
            node.score = 0
    
    else: #if loop is unbroken
        print(f"Convergence not reached; stopped at {rounds} iterations")
        print(f"Cooperators: {strategies.count('C')}, Defectors: {strategies.count('D')}, graph changed: {switches_occurred}")
        if draw:
            draw_graph(G)
    if draw:        
        coop_score_hist = [avg_score_history[i]['C'] for i in range(len(avg_score_history))]
        defect_score_hist = [avg_score_history[i]['D'] for i in range(len(avg_score_history))]
        plt.figure(figsize = (8, 8))
        plt.plot(coop_score_hist, label = 'average cooperator score')
        plt.plot(defect_score_hist, label = 'average defector score')
        plt.xlabel('round')
        plt.ylabel('average score')
        plt.title('Score over time')
        plt.legend()
        
        plt.figure(figsize = (8, 8))
        plt.plot(C_pop_hist, label = 'cooperator population')
        plt.plot(D_pop_hist, label = 'defector population')
        plt.xlabel('round')
        plt.ylabel('population')
        plt.title('Population over time')
        plt.legend()
    return G

def run_grid_search(G, rounds, coop_bonus_range, defector_bonus_range, sucker_loss_range, resolution = 5, draw = False, output_dir = None):
    '''
    Performs a grid search of score matrix parameters, to explore graph behavior for different scoring situations. Function automatically makes a
    deep copy of the original graph, so that the behavior can be explored for the same initial conditions. 

    score_matrix = {'DD' : 0, 'DC' : 1 + defector_bonus, 'CD' : -sucker_loss, 'CC' : 1 + coop_bonus}

    Args:
        G : the graph to be grid searched.
        rounds : number of rounds for each simulation grid point.
        
        coop_bonus_range : a tuple containing the start and end point for the coop_bonus grid search, 
        i.e. (coop_bonus_initial, coop_bonus_final)
        defector_bonus_range : a tuple containing the start and end point for the defector_bonus grid search, 
        i.e. (defector_bonus_initial, defector_bonus_final)
        sucker_loss_range : a tuple containing the start and end point for the sucker_loss grid search, 
        i.e. (sucker_loss_initial, sucker_loss_final)

        resolution : determines the resolution for the grid_search. All 3 grid parameters have the same resolution, so the number of grid points
        is equal to resolution**3. Default value of 5 corresponds to 125 grid points. 
        draw : if true, will draw graphs when relevant.
        output_dir : if true, will save all relevant figures to the output directory.

    Outputs:
        returns a dictionary of tuple : networkx Graph pairs, where the tuple contains the grid coordinates based on the 
        (defector, sucker, cooperator) bonuses, and the graph is the final graph.
    
    '''
    assert len(coop_bonus_range) == len(defector_bonus_range) == len(sucker_loss_range) == 2, 'ranges improperly defined, lens != 2'
    assert isinstance(resolution, int), 'resolution is not an integer'

    output_graphs = {}
    #GRID CONSTRUCTION
    coop_bonus = np.linspace(coop_bonus_range[0], coop_bonus_range[1], resolution)
    defector_bonus = np.linspace(defector_bonus_range[0], defector_bonus_range[1], resolution)
    sucker_loss = np.linspace(sucker_loss_range[0], sucker_loss_range[1], resolution)
    
    #GRID SEARCH
    for d in defector_bonus:
        for s in sucker_loss:
            for c in coop_bonus:
                score_dic = {"DD": 0, "DC": 1 + d, "CD": -s, "CC": 1 + c}
                G_out = run_simulation(copy.deepcopy(G), rounds, score_dic, draw=draw)
                output_graphs[(d,s,c)] = G_out
                if draw:
                    print(f'SCORE MATRIX : {score_dic}')
                    draw_graph(G_out)
    return output_graphs

def grid_search_plot(G, grid_graph_dict, coop_bonus_range, defector_bonus_range, sucker_loss_range, resolution):

    #GRID CONSTRUCTION
    coop_bonus = np.linspace(coop_bonus_range[0], coop_bonus_range[1], resolution)
    defector_bonus = np.linspace(defector_bonus_range[0], defector_bonus_range[1], resolution)
    sucker_loss = np.linspace(sucker_loss_range[0], sucker_loss_range[1], resolution)

    num_nodes = G.number_of_nodes()
    init_strats = [node.strategy for node in G.nodes]
    idx_d = 0
    idx_s = 0
    idx_c = 0

    coop_pop = np.zeros((resolution, resolution, resolution))
    
    for d in defector_bonus:
        idx_s = 0
        for s in sucker_loss:
            idx_c = 0
            for c in coop_bonus:
                cur_G = grid_graph_dict[(d,s,c)]
                strats = [node.strategy for node in cur_G.nodes]
                coop_pop[idx_d, idx_s, idx_c] = strats.count('C') / num_nodes
                idx_c += 1
            idx_s += 1
        idx_d += 1

    #MAKING THE VOXEL PLOT
    ## Voxel Dimensions
    dd = (defector_bonus[1] - defector_bonus[0]) * 0.75
    ds = (sucker_loss[1] - sucker_loss[0]) * 0.75
    dc = (coop_bonus[1] - coop_bonus[0]) * 0.75

    filled = np.ones_like(coop_pop, dtype=bool)  # Plot all voxels

    # Sub function to map the cooperator population ratio to colors.
    # if cooperator pop == 1/2 node population, color's gray
    # if cooperator pop == node population, color's green
    # if cooperator pop == 0, color's red
    def val_to_rgb(v):
        if v < 0.5:
            t = v / 0.5
            return (1 * (1 - t) + 0.5 * t, 0 * (1 - t) + 0.5 * t, 0 * (1 - t) + 0.5 * t)  # red to gray
        else:
            t = (v - 0.5) / 0.5
            return (0.5 * (1 - t) + 0 * t, 0.5 * (1 - t) + 1 * t, 0.5 * (1 - t) + 0 * t)  # gray to green

    # Map each voxel value to RGB
    colors = np.empty(coop_pop.shape + (4,), dtype=float)  # RGBA
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                rgb = val_to_rgb(coop_pop[i, j, k])
                colors[i, j, k] = (*rgb, 0.3)  # alpha = 0.3

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(filled, facecolors=colors, edgecolors='k', linewidth=0.3)
    # Set axis ticks and labels to match actual coordinate values
    ax.set_xticks(np.linspace(0.5, resolution - 0.5, 5))  # 5 major ticks
    ax.set_yticks(np.linspace(0.5, resolution - 0.5, 5))
    ax.set_zticks(np.linspace(0.5, resolution - 0.5, 5))
    
    ax.set_xticklabels(np.linspace(0, defector_bonus_range[1], 5).round(2))
    ax.set_yticklabels(np.linspace(0, sucker_loss_range[1], 5).round(2))
    ax.set_zticklabels(np.linspace(0, coop_bonus_range[1], 5).round(2))

    ax.set_box_aspect((defector_bonus_range[1], sucker_loss_range[1], coop_bonus_range[1]))
    ax.set_xlabel('defector bonus')
    ax.set_ylabel('sucker loss')
    ax.set_zlabel('cooperator bonus')
    plt.tight_layout()
    plt.show()

def analyze_network_properties(G):
    """
    Analyze and print basic properties of a network G.
    """
    print(f"\nNetwork Analysis:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    print(f"Clustering coefficient: {nx.average_clustering(G):.3f}")
    print(f"Number of connected components: {nx.number_connected_components(G)}")

    if nx.is_connected(G):
        print(f"Average shortest path length: {nx.average_shortest_path_length(G):.2f}")
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        subG = G.subgraph(largest_cc)
        print(f"Average shortest path length (largest component): {nx.average_shortest_path_length(subG):.2f}")

    #Node degree distribution plot
    degree_hist = nx.degree_histogram(G)
    degrees = range(len(degree_hist))
    plt.figure(figsize = (8,8))
    plt.loglog(degrees, degree_hist)
    plt.xlabel('degree')
    plt.ylabel('frequency')







###
# FACEBOOK STUFF
###



def draw_influencer_graph(G, influencers, round_num=None):
    """
    Draw the graph G, highlighting influencer nodes with a golden edge.
    This will run very very slowly for large N because of the position determination.
    
    Args:
      G: NetworkX graph with Player nodes having .strategy attribute
      influencers: list of influencer nodes to highlight
      round_num: optional int, current round number for title
    """
    pos = nx.spring_layout(G, seed=42)  # consistent layout
    
    # Colors by strategy
    color_map = {'C': 'green', 'D': 'red'}
    node_colors = [color_map.get(node.strategy, 'gray') for node in G.nodes]

    node_size = 10  # small size for all nodes

    # Draw all nodes first with no edge color
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size, alpha=0.7)

    # Overlay influencers with golden edge but same size
    nx.draw_networkx_nodes(G, pos, nodelist=influencers,
                           node_color=[color_map.get(n.strategy, 'gray') for n in influencers],
                           node_size=node_size, edgecolors='gold', linewidths=2)

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.4)

    plt.title(f'Network at Round {round_num}' if round_num is not None else 'Network')
    plt.axis('off')
    plt.show()



def run_influencer_experiment(G, rounds=50, threshold=None, alpha=1.0, bonus=1.0):
    """
    tries to run simulation while setting influencers to all the same category.
    not sure if this works yet.
    """
    degrees = dict(G.degree())
    if threshold is None:
        deg_values = list(degrees.values())
        threshold = sorted(deg_values)[int(0.97 * len(deg_values))]

    # influencers and non-influencers are Player objects directly
    influencers = [n for n, deg in degrees.items() if deg > threshold]
    for node in G.nodes():
        if node in influencers:
            node.strategy = 'D'

    print(f"Identified {len(influencers)} influencer nodes with degree > {threshold}")

    expected_degree = sum(degrees.values()) / len(degrees)

    def count_strategies():
        coop = sum(1 for n in G.nodes if n.strategy == 'C')
        defect = G.number_of_nodes() - coop
        return coop, defect

    results = {
        'coop_influencers': [],
        'defect_influencers': [],
        'coop_counts': [],
        'defect_counts': []
    }