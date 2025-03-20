import numpy as np
import random
from math import cos, sin, pi
from scipy.spatial.distance import pdist, squareform

data = np.genfromtxt('TSP.csv', delimiter=',', skip_header=1)

POP_SIZE = 100
GENS = 600
MUT_RATE = 0.1

def get_customers_for_env(e):
    num_customers = 50 + 10 * e
    adjusted_data = data[:num_customers, :].copy()
    if e != 0:
        factor = 2 * e
        adjusted_data[:, 1] += factor * np.cos(pi / (2 * e))
        adjusted_data[:, 2] += factor * np.sin(pi / (2 * e))
    return adjusted_data

def calculate_distance_matrix(customers):
    coords = customers[:, 1:3]
    return squareform(pdist(coords))

def total_distance(route, distance_matrix):
    return sum(distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1)) + distance_matrix[route[-1], route[0]]

# initial population
def create_initial_population(size, num_customers):
    return [random.sample(range(num_customers), num_customers) for _ in range(size)]

def select(population, distance_matrix):
    tournament = random.sample(population, 5)
    tournament.sort(key=lambda route: total_distance(route, distance_matrix))
    return tournament[0]

# Crossover
def crossover(parent1, parent2):
    if random.random() < 0.8:
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size

        # Copy from parent1
        child[start:end] = parent1[start:end]

        # Fill from parent2
        p2_index = 0
        for i in range(size):
            if child[i] is None:
                while parent2[p2_index] in child:
                    p2_index += 1
                child[i] = parent2[p2_index]
                p2_index += 1

        return child
    else:
        return parent1[:]

# Mutation
def mutate(route):
    if random.random() < MUT_RATE:
        idx1, idx2 = random.sample(range(len(route)), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]

# Genetic Algorithm
def genetic_algorithm(customers, reuse_previous=False, previous_population=None):
    num_customers = len(customers)
    distance_matrix = calculate_distance_matrix(customers)
    
    if reuse_previous and previous_population:
        population = [ind + list(range(len(ind), num_customers)) for ind in previous_population]
        population += create_initial_population(POP_SIZE - len(population), num_customers)
    else:
        population = create_initial_population(POP_SIZE, num_customers)
    
    for generation in range(GENS):
        fitnesses = [total_distance(ind, distance_matrix) for ind in population]
        new_population = []
        for _ in range(POP_SIZE):
            parent1 = select(population, distance_matrix)
            parent2 = select(population, distance_matrix)
            child = crossover(parent1, parent2)
            mutate(child)
            new_population.append(child)
        population = new_population

    best_index = np.argmin([total_distance(ind, distance_matrix) for ind in population])
    return population[best_index], total_distance(population[best_index], distance_matrix)

def main():
    previous_population = None
    for e in range(6):
        customers = get_customers_for_env(e)
        best_route, best_distance = genetic_algorithm(customers, reuse_previous=False)
        print(f"Environment {e} without reuse: Best Distance = {best_distance}")
        
        if previous_population is not None:
            best_route_reuse, best_distance_reuse = genetic_algorithm(customers, reuse_previous=True, previous_population=previous_population)
            print(f"Environment {e} with reuse: Best Distance = {best_distance_reuse}")

        previous_population = [best_route[i:i+50+10*e] for i in range(0, len(best_route), 50+10*e)]

if __name__ == "__main__":
    main()