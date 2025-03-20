import numpy as np
import pandas as pd
import math

# 读取数据
data = pd.read_csv('TSP.csv')

# 客户之间的欧几里得距离矩阵
def calculate_distance_matrix(data):
    num_customers = len(data)
    distance_matrix = np.zeros((num_customers, num_customers))
    for i in range(num_customers):
        for j in range(num_customers):
            if i!= j:
                x1, y1 = data.loc[i, '  XCOORD.   '], data.loc[i, 'YCOORD.']
                x2, y2 = data.loc[j, '  XCOORD.   '], data.loc[j, 'YCOORD.']
                distance_matrix[i][j] = math.sqrt((x2 - x1)**2 + (x2 - y1)**2)
    return distance_matrix

distance_matrix = calculate_distance_matrix(data)

# 总距离
def total_travel_distance(route, distance_matrix):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i]][route[i + 1]]
    total_distance += distance_matrix[route[-1]][route[0]]
    return total_distance

# 总利润
def total_sales_profit(route, data, distance_matrix):
    total_profit = 0
    for i in range(len(route)):
        customer_profit = data.loc[route[i], 'PROFIT']
        if i > 0:
            travel_time = distance_matrix[route[i - 1]][route[i]]
            customer_profit -= travel_time
        total_profit += customer_profit
    return total_profit

# 基于帕累托优势选择的方法
def pareto_dominance_method(data, distance_matrix, num_iterations):
    
    solutions = []
    for _ in range(num_iterations):
        route = np.random.permutation(len(data))
        f1 = total_travel_distance(route, distance_matrix)
        f2 = total_sales_profit(route, data, distance_matrix)
        is_dominated = False
        for other_solution in solutions:
            f1_other = total_travel_distance(other_solution, distance_matrix)
            f2_other = total_sales_profit(other_solution, data, distance_matrix)
            if (f1_other <= f1 and f2_other > f2) or (f1_other < f1 and f2_other >= f2):
                is_dominated = True
                break
        if not is_dominated:
            solutions.append(route)
    return solutions


num_iterations = 1000

optimal_solutions = pareto_dominance_method(data, distance_matrix, num_iterations)

for solution in optimal_solutions:
    f1 = total_travel_distance(solution, distance_matrix)
    f2 = total_sales_profit(solution, data, distance_matrix)
    print("非支配解路线：", solution)
    print("总旅行距离：", f1)
    print("总销售利润：", f2)