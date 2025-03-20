import pandas as pd
import numpy as np
import math
import random
from deap import base, creator, tools, algorithms

data = pd.read_csv('TSP.csv')
customers = data[['CUST NO. ', '  XCOORD.   ', 'YCOORD.']].values
profits = data['PROFIT'].values
ready_times = data['READY TIME'].values
due_times = data['DUE TIME'].values


def calculate_distance_matrix(data):
    num_customers = len(data)
    distance_matrix = np.zeros((num_customers, num_customers))
    for i in range(num_customers):
        for j in range(num_customers):
            if i!= j:
                x1, y1 = data.loc[i, '  XCOORD.   '], data.loc[i, 'YCOORD.']
                x2, y2 = data.loc[j, '  XCOORD.   '], data.loc[j, 'YCOORD.']
                distance_matrix[i][j] = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance_matrix

distance_matrix = calculate_distance_matrix(data)

creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(len(customers)), len(customers))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual, 100)  # 种群大小为100

def evaluate(individual):
    total_distance = 0
    total_profit = 0
    total_violation = 0
    for i in range(len(individual)):
        if i == 0:
            total_distance += distance_matrix[0][individual[i]]
        else:
            total_distance += distance_matrix[individual[i - 1]][individual[i]]
        arrival_time = total_distance
        if arrival_time < ready_times[individual[i]]:
            total_violation += ready_times[individual[i]] - arrival_time
        elif arrival_time > due_times[individual[i]]:
            total_violation += arrival_time - due_times[individual[i]]
        if i == 0:
            total_profit += profits[individual[i]] - distance_matrix[0][individual[i]]
        else:
            total_profit += profits[individual[i]] - distance_matrix[individual[i - 1]][individual[i]]

    return total_distance, total_profit, total_violation

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selNSGA2)

def main():
    pop = toolbox.population()
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaMuPlusLambda(pop, toolbox, mu=50, lambda_=100, cxpb=0.7, mutpb=0.3, ngen=100, stats=stats, halloffame=hof, verbose=False)
    return pop, hof, stats

if __name__ == "__main__":
    pop, hof, stats = main()
    for ind in hof:
        total_distance, total_profit, total_violation = ind.fitness.values
        print("----------")
        print("总距离：", total_distance)
        print("总利润：", total_profit)
        print("总违反值：", total_violation)