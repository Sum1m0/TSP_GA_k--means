import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import random

data = pd.read_csv('TSP.csv')

# 添加新顾客
new_customers = data.copy()
new_customers['  XCOORD.   '] += 100
data = pd.concat([data, new_customers], ignore_index=True)

customers = data[['  XCOORD.   ', 'YCOORD.']].values

# 聚类5个区域
kmeans = KMeans(n_clusters=5, random_state=0).fit(customers)
labels = kmeans.labels_

# 顾客数据按照聚类分组
clustered_data = [data[labels == i] for i in range(5)]

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

        current_best = min(population, key=lambda route: total_distance(route))
        if total_distance(current_best) < total_distance(best_route):
            best_route = current_best

    return best_route

# 遗传算法求解最优路径
optimal_paths = []
for cluster in clustered_data:
    num_customers_in_cluster = len(cluster)
    ga_result = genetic_algorithm(pop_size=100, num_customers=num_customers_in_cluster, generations=100)
    optimal_paths.append(ga_result)

# 计算总路径长度
total_path_length = 0
for path, cluster in zip(optimal_paths, clustered_data):
    total_path_length += total_distance(path)

print("最终路程:", total_path_length)

# 最终路程: 2229.279768968941