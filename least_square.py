""" Docstring:
Description: weighted least square
Author: Henry Zhang
Date:June 26, 2019
"""

# module
import numpy as np
from matplotlib import pyplot as plt
# classes

# functions
def prepareDataMatrix(dataMatrix):
  A = 1.0
  B = 1.0
  sum_i = 0
  for i in range(dataMatrix.shape[0]):
    sum_i += i/1000 * np.log10(i+1)
    dataMatrix[i,0] = sum_i
    dataMatrix[i,1] = A * sum_i + B - 2*(dataMatrix.shape[0] - i) / 50
  dataMatrix += np.random.randn(dataMatrix.shape[0], dataMatrix.shape[1]) * dataMatrix.shape[0] / 10000 / (5 - np.log(dataMatrix.shape[0])) 

def fit_ls_loop(dataMatrix, dataHomo):
  sum_x = 0
  sum_y = 0
  sum_xy = 0
  sum_xx = 0
  point_size = dataMatrix.shape[0]
  for i in range(point_size):
    sum_x += dataMatrix[i,0]
    sum_y += dataMatrix[i,1]
    sum_xy += dataMatrix[i,0] * dataMatrix[i,1]
    sum_xx += dataMatrix[i,0] * dataMatrix[i,0]
  a = (point_size * sum_xy - sum_x * sum_y) / (point_size * sum_xx - sum_x * sum_x)
  b = (sum_y - a * sum_x) / point_size
  line_param = np.array([a,b])
  diff = np.abs( np.matmul(dataMatrix, np.array([[1, line_param[0]]]).T ) - line_param[1] )
  error = np.sum( diff) / point_size / (line_param[0] ** 2 + 1 )
  return line_param, error

def compute_weight(dataMatrix, dataHomo, line_param):
  # compute the distance
  direction = np.array([[1, line_param[0]]]).T
  proj_vector = np.matmul(dataMatrix, direction)
  proj_vector = np.sort(proj_vector)
  distance_vector = proj_vector[1:] - proj_vector[0:-1] 
  weight = np.zeros(proj_vector.shape[0])
  weight[0:-1] += distance_vector[0:,0]
  weight[1:] += distance_vector[0:,0]
  weight[0] += distance_vector[0]
  weight[-1] += distance_vector[-1]
  weight = weight ** 2
  weight_normed = weight / np.sum(weight) # np.linalg.norm(weight)
  return weight_normed



def fit_ls_weighted(dataMatrix, dataHomo, line_param):
  weight = compute_weight(dataMatrix, dataHomo, line_param)
  sum_wx = 0
  sum_wy = 0
  sum_wxy = 0
  sum_wxx = 0
  sum_w = 0
  point_size = dataMatrix.shape[0]
  for i in range(point_size):
    sum_wx += weight[i] * dataMatrix[i,0]
    sum_wy += weight[i] * dataMatrix[i,1]
    sum_wxy += weight[i] * dataMatrix[i,0] * dataMatrix[i,1]
    sum_wxx += weight[i] * dataMatrix[i,0] * dataMatrix[i,0]
    sum_w += weight[i]
  a = (sum_w * sum_wxy - sum_wx * sum_wy) / (sum_w * sum_wxx - sum_wx * sum_wx)
  b = (sum_wy - a * sum_wx) / sum_w
  line_param_weighted = np.array([a,b])
  diff_weighted = np.abs( np.matmul( dataMatrix, np.array( [[1, line_param_weighted[0]]] ).T ) - line_param_weighted[1] ) * weight / sum_w
  error_weighted = np.sum(diff_weighted) / point_size / (line_param_weighted[0] ** 2 + 1 )
  return line_param_weighted, error_weighted

def vis(dataMatrix, line_param, line_param_weighted):
  plt.figure(figsize=(16,8))

  plt.subplot("121")
  plt.scatter(dataMatrix[:,0], dataMatrix[:,1], s=1)
  x = np.linspace(min(dataMatrix[:,0]), max(dataMatrix[:,0]), 2)
  y = line_param[0] * x + line_param[1] 
  plt.plot(x,y, c='r')
  plt.title("equal weight line fit")

  plt.subplot("122")
  plt.scatter(dataMatrix[:,0], dataMatrix[:,1], s=1)
  y_weighted = line_param_weighted[0] * x + line_param_weighted[1]
  plt.plot(x,y_weighted, c='r')
  plt.title("weighted line fit")

  plt.show(block=True)

# main
def main():
  ROW = 100
  COL = 2
  dataMatrix = np.zeros([ROW, COL])
  prepareDataMatrix(dataMatrix)
  # assert(dataMatrix[3,0] == 3)

  dataHomo = np.ones([ROW,1])
  line_param, error = fit_ls_loop(dataMatrix, dataHomo)
  # line_param_ab, error, rank, singular_value = np.linalg.lstsq(dataMatrix, dataHomo)
  # line_slope = -1 * line_param_ab[0] / line_param_ab[1]
  # line_intercept = 1 / line_param_ab[1]
  # line_param = np.array([line_slope, line_intercept])
  print("error", error)
  
  line_param_weighted, error_weighted = fit_ls_weighted(dataMatrix, dataHomo, line_param)
  print("error_weighted", error_weighted)
  vis(dataMatrix, line_param, line_param_weighted)

if __name__ == "__main__":
  main()