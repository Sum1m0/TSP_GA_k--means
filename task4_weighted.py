import numpy as np
import pandas as pd
import math

data = pd.read_csv('TSP.csv')

# 距离矩阵
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

# 加权
def weighted_objective_function(route, data, distance_matrix, lambda_value):
    f1 = total_travel_distance(route, distance_matrix)
    f2 = total_sales_profit(route, data, distance_matrix)
    return f1 - lambda_value * f2

def find_optimal_solution(data, distance_matrix, lambda_value):
    best_route = None
    best_value = np.inf
    for _ in range(1000):  
        route = np.random.permutation(len(data))
        value = weighted_objective_function(route, data, distance_matrix, lambda_value)
        if value < best_value:
            best_value = value
            best_route = route
    return best_route

lambda_value = 1.5

optimal_route = find_optimal_solution(data, distance_matrix, lambda_value)
optimal_f1 = total_travel_distance(optimal_route, distance_matrix)
optimal_f2 = total_sales_profit(optimal_route, data, distance_matrix)
print("基于加权目标函数的优化结果：")
print("最佳路线：", optimal_route)
print("总旅行距离：", optimal_f1)
print("总销售利润：", optimal_f2)