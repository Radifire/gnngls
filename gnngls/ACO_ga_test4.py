import itertools
import numpy as np
import networkx as nx

def generate_instances(n_nodes, n_instances=1):
    G = nx.Graph()

    coords = np.random.random((n_nodes, 2))
    for n, p in enumerate(coords):
        G.add_node(n, pos=p)

    for i, j in itertools.combinations(G.nodes, 2):
        w = np.linalg.norm(G.nodes[j]['pos'] - G.nodes[i]['pos'])
        G.add_edge(i, j, weight=w)

    return G

def tsp_aco(G, n_ants=10, n_iterations=100, alpha=1, beta=3, evaporation_rate=0.5,
            mutation_rate=0.1, mutation_period=10, mutation_size=3):
    n_nodes = G.number_of_nodes()
    best_solution = None
    best_fitness = np.inf
    times_consecutive_unchanged = 0
    pheromones = np.ones((n_nodes, n_nodes))

    def fitness(tour):
        return sum(G[u][v]['weight'] for u, v in zip(tour[:-1], tour[1:]))

    def prob(i, j, tour):
        pheromone = pheromones[i][j]
        distance = G[i][j]['weight']
        unvisited = [n for n in range(n_nodes) if n not in tour]
        if not unvisited:
            return 0
        numerator = pheromone ** alpha * (1 / distance) ** beta
        denominator = sum(pheromones[i][v] ** alpha * (1 / G[i][v]['weight']) ** beta for v in unvisited)
        return numerator / denominator

    for iteration in range(n_iterations):
        solutions = []
        for ant in range(n_ants):
            start_node = np.random.choice(n_nodes)
            tour = [start_node]
            for _ in range(n_nodes - 1):
                probs = [prob(tour[-1], j, tour) for j in range(n_nodes) if j not in tour]
                next_node = np.random.choice([j for j in range(n_nodes) if j not in tour], p=probs)
                tour.append(next_node)
            solutions.append(tour)
            ant_fitness = fitness(tour)
            if ant_fitness < best_fitness:
                best_fitness = ant_fitness
                best_solution = tour
                pheromones *= (1 - evaporation_rate)
                for i, j in zip(tour[:-1], tour[1:]):
                    pheromones[i][j] += 1 / ant_fitness
        for i, j in itertools.combinations(range(n_nodes), 2):
            pheromones[i][j] *= (1 - evaporation_rate)
        if best_solution is not None and iteration - times_consecutive_unchanged >= mutation_period:
            for _ in range(mutation_size):
                i, j = np.random.choice(n_nodes, size=2, replace=False)
                pheromones[i][j] = pheromones[j][i] = np.random.rand()
            times_consecutive_unchanged = 0
            best_solution = None
        else:
            times_consecutive_unchanged += 1

    return best_solution, best_fitness

n_nodes = 20
G = generate_instances(n_nodes)

print(tsp_aco(G, n_iterations=100))
print(tsp_aco(G, n_iterations=100))
print(tsp_aco(G, n_iterations=100))
print(tsp_aco(G, n_iterations=100))
print(tsp_aco(G, n_iterations=100))