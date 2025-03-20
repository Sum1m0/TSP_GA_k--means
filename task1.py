import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

data = pd.read_csv('TSP.csv')
customers = data[['CUST NO. ', '  XCOORD.   ', 'YCOORD.']].values

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def total_distance(route):
    distance = 0
    for i in range(len(route)):
        distance += calculate_distance(customers[route[i]], customers[route[(i + 1) % len(route)]])
    return distance

def create_initial_population(pop_size, num_customers):
    return [random.sample(range(num_customers), num_customers) for _ in range(pop_size)]

def select(population):
    tournament = random.sample(population, 5)
    tournament.sort(key=lambda route: total_distance(route))
    return tournament[0]

def crossover(parent1, parent2):
    if random.random() < 0.8:
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size

        #parent1
        child[start:end] = parent1[start:end]

        #parent2
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
    
def mutate(route):
    if random.random() < 0.1: 
        idx1, idx2 = random.sample(range(len(route)), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]  # swap

def genetic_algorithm(pop_size, num_customers, generations):
    population = create_initial_population(pop_size, num_customers)
    best_route = min(population, key=lambda route: total_distance(route))

    for _ in range(generations):
        new_population = []
        for _ in range(pop_size):
            parent1 = select(population)
            parent2 = select(population)
            child = crossover(parent1, parent2)
            mutate(child)
            new_population.append(child)
        population = new_population

        # Update best route
        current_best = min(population, key=lambda route: total_distance(route))
        if total_distance(current_best) < total_distance(best_route):
            best_route = current_best

    return best_route

pop_size = 200
num_customers = len(customers)
generations = 1000

best_route = genetic_algorithm(pop_size, num_customers, generations)

best_distance = total_distance(best_route)

best_route_coordinates = customers[best_route]
plt.figure(figsize=(10, 6))
plt.plot(best_route_coordinates[:, 0], best_route_coordinates[:, 1], marker='o')
plt.plot([best_route_coordinates[0, 0], best_route_coordinates[-1, 0]],
         [best_route_coordinates[0, 1], best_route_coordinates[-1, 1]], 'ro') 
plt.title('Best Route Found')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid()
plt.show()

print(f'Total distance of the best route: {best_distance}')
