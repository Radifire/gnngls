import time
import random
import networkx as nx

def ant_colony_optimization(graph, weight='regret_pred', guides='weight', num_ants=10, alpha=1, beta=2, evaporation_rate=0.5, pheromone_constant=1, time_limit=10):
    # Initialize pheromone levels on each edge
    pheromone_levels = {}
    for u, v in graph.edges():
        pheromone_levels[(u, v)] = 1
    
    # Initialize variables to keep track of the best tour found
    best_tour = None
    best_tour_length = float('inf')
    
    # Keep track of the start time
    start_time = time.time()
    
    # Repeat until the time limit is reached
    while time.time() - start_time < time_limit:
        # Place each ant on a random starting node
        ants = [random.choice(list(graph.nodes())) for _ in range(num_ants)]
        
        # Initialize visited nodes for each ant
        ant_distances = {ant: 0 for ant in ants}
        ant_visited = {ant: set([ant]) for ant in ants}
        
        # Move each ant to a new node based on the pheromone levels and edge weights
        for _ in range(graph.number_of_nodes()-1):
            # Calculate the probability of moving to each neighboring node for each ant in parallel
            unvisited_neighbors = [set(graph.neighbors(ant)) - ant_visited[ant] for ant in ants]
            probabilities = []
            for ant in ants:
                current_node = ant_visited[ant][-1]
                ant_probabilities = {}
                denominator = 0
                for neighbor in unvisited_neighbors[ant]:
                    edge_weight_value = graph[current_node][neighbor][weight]
                    pheromone_level = pheromone_levels[(current_node, neighbor)]
                    numerator = pheromone_level ** alpha * (1/edge_weight_value) ** beta
                    denominator += numerator
                    ant_probabilities[neighbor] = numerator
                for neighbor in ant_probabilities:
                    ant_probabilities[neighbor] /= denominator
                probabilities.append(ant_probabilities)
            
            # Select a neighbor to move to for each ant in parallel based on the probabilities
            selected_neighbors = [random.choices(list(prob.keys()), list(prob.values()))[0] for prob in probabilities]
            for ant, neighbor in zip(ants, selected_neighbors):
                ant_visited[ant].add(neighbor)
                
        # Update the pheromone levels on each edge based on the distances traveled by each ant
        for u, v in graph.edges():
            delta_pheromone = 0
            for ant in ants:
                if v in ant_visited[ant] and u in ant_visited[ant]:
                    delta_pheromone += pheromone_constant / ant_distances[ant]
            pheromone_levels[(u, v)] = (1 - evaporation_rate) * pheromone_levels[(u, v)] + delta_pheromone
            
        # Determine the best tour found in this iteration
        iteration_best_tour = None
        iteration_best_tour_length = float('inf')
        for ant in ants:
            tour_length = ant_distances[ant] + graph[ant_visited[ant][-1]][ant_visited[ant][0]][weight]
            if tour_length < iteration_best_tour_length:
                iteration_best_tour = list(ant_visited[ant])
                iteration_best_tour_length = tour_length
        
        # Update the best tour if necessary
        if iteration_best_tour_length < best_tour_length:
            best_tour = iteration_best_tour
            best_tour_length = iteration_best_tour_length
    
    return best_tour, best_tour_length
