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
def prepareDataMatrix(dataMatrix, data_model = "line_curved"):
  if "line_curved" == data_model:
    dataMatrix = create_curved_line(dataMatrix)
  if "corner" == data_model:
    dataMatrix = create_corner(dataMatrix)
  return dataMatrix

def create_curved_line(dataMatrix):
  A = 1.0
  B = 1.0
  sum_i = 0
  for i in range(dataMatrix.shape[0]):
    sum_i += i/1000 * np.log10(i+1)
    dataMatrix[i,0] = sum_i
    dataMatrix[i,1] = A * sum_i + B - 2*(dataMatrix.shape[0] - i) / 50
  dataMatrix += np.random.randn(dataMatrix.shape[0], dataMatrix.shape[1]) * dataMatrix.shape[0] / 10000 / (5 - np.log(dataMatrix.shape[0])) 
  dataMatrix = np.random.permutation(dataMatrix)
  return dataMatrix

def create_corner(dataMatrix):
  A = -1.0
  B = 100.0
  scale_factor = 6
  sep_point = int(dataMatrix.shape[0] / 3)
  for i in range(dataMatrix.shape[0]):
    px = i / scale_factor
    dataMatrix[i, 0] = px
    if i < sep_point:
      dataMatrix[i, 1] = A * px + B
    else:
      dataMatrix[i, 1] = -1 / A * px + (A - -1/A) * sep_point/scale_factor + B
  dataMatrix += np.random.randn(dataMatrix.shape[0], dataMatrix.shape[1]) / scale_factor / 5
  return dataMatrix
  

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
  weight_index = np.argsort(proj_vector[:,0])
  proj_vector = np.sort(proj_vector, axis=0)
  distance_vector = proj_vector[1:] - proj_vector[0:-1] 
  weight = np.zeros(proj_vector.shape[0])
  weight[0:-1] += distance_vector[0:,0]
  weight[1:] += distance_vector[0:,0]
  weight[0] += distance_vector[0]
  weight[-1] += distance_vector[-1]
  weight = weight ** 2
  weight_normed = weight / np.sum(weight) # np.linalg.norm(weight)
  return weight_normed, weight_index

def reject_outliers(dataMatrix, dataHomo, line_param_weighted, weight):
  # sum_w = np.sum(weight)
  diff = np.abs( np.matmul( dataMatrix, np.array( [[line_param_weighted[0], -1]] ).T ) + line_param_weighted[1] ).reshape([-1])
  diff_index = np.argsort(diff)[0:int(0.8*dataHomo.shape[0])]
  return dataMatrix[diff_index,:], dataHomo[diff_index,:]

def fit_line(dataMatrix, dataHomo, vis = True):
  line_param, error = fit_ls_loop(dataMatrix, dataHomo)
  weight, weight_index = compute_weight(dataMatrix, dataHomo, line_param)
  line_param_weighted, error_weighted = fit_ls_weighted(dataMatrix, dataHomo, weight, weight_index)
  dataMatrixPart, dataHomoPart = reject_outliers(dataMatrix, dataHomo, line_param_weighted, weight)
  weight_part, weight_index_part = compute_weight(dataMatrixPart, dataHomoPart, line_param_weighted)
  line_param_final, error_final = fit_ls_weighted(dataMatrixPart, dataHomoPart, weight_part, weight_index_part)
  if vis:
    vis_data_line_1(dataMatrix, line_param)
    vis_data_line_2(dataMatrix, line_param_weighted)
    vis_data_line_3(dataMatrix, line_param_final, dataMatrixPart)
  return line_param_final, error_final

def fit_ls_weighted(dataMatrix, dataHomo, weight, weight_index):
  sum_wx = 0
  sum_wy = 0
  sum_wxy = 0
  sum_wxx = 0
  sum_w = 0
  point_size = dataMatrix.shape[0]
  for i in range(point_size):
    dx = dataMatrix[weight_index[i],0]
    dy = dataMatrix[weight_index[i],1]
    sum_wx += weight[i] * dx
    sum_wy += weight[i] * dy
    sum_wxy += weight[i] * dx * dy
    sum_wxx += weight[i] * dx * dx
    sum_w += weight[i]
  a = (sum_w * sum_wxy - sum_wx * sum_wy) / (sum_w * sum_wxx - sum_wx * sum_wx)
  b = (sum_wy - a * sum_wx) / sum_w
  line_param_weighted = np.array([a,b])
  diff_weighted = np.abs( np.matmul( dataMatrix, np.array( [[line_param_weighted[0], -1]] ).T ) + line_param_weighted[1] ).reshape([-1]) * weight / sum_w
  error_weighted = np.sum(diff_weighted) / point_size / (line_param_weighted[0] ** 2 + 1 )
  return line_param_weighted, error_weighted

def split_data(dataMatrix, dataHomo, line_param):
  diff_thred = 0.1
  diff = np.abs( np.matmul( dataMatrix, np.array( [[line_param[0], -1]] ).T ) + line_param[1] ).reshape([-1])
  
  diff_sorted = np.sort(diff)
  sep_point = diff.shape[0]
  for i in range(diff.shape[0]):
    if diff_sorted[i] > diff_thred:
      sep_point = i
      break
  print("sep_point: ", sep_point)
  diff_index = np.argsort(diff)[sep_point:]

  return dataMatrix[diff_index,:], dataHomo[diff_index,:]

def fit_corner(dataMatrix, dataHomo, line_param):
  dataMatrixPart, dataHomoPart = split_data(dataMatrix, dataHomo, line_param)
  line_param_other, error_other = fit_line(dataMatrixPart, dataHomoPart, vis=False)
  if line_param[0] * line_param_other[0] > 0:
    print("error")
  vis_corner(dataMatrixPart, line_param_other)


def vis_init():
  plt.figure(figsize=(16,6))

def vis_data_line_1(dataMatrix, line_param):
  plt.subplot("131")
  plt.scatter(dataMatrix[:,0], dataMatrix[:,1], s=1)
  x = np.linspace(min(dataMatrix[:,0]), max(dataMatrix[:,0]), 2)
  y = line_param[0] * x + line_param[1] 
  plt.plot(x,y, c='r')
  plt.axis('scaled')
  plt.title("equal weight line fit")

  plt.pause(0.01)

def vis_data_line_2(dataMatrix, line_param_weighted):
  plt.subplot("132")
  plt.scatter(dataMatrix[:,0], dataMatrix[:,1], s=1)
  x = np.linspace(min(dataMatrix[:,0]), max(dataMatrix[:,0]), 2)
  y_weighted = line_param_weighted[0] * x + line_param_weighted[1]
  plt.plot(x,y_weighted, c='r')
  plt.axis('scaled')
  plt.title("weighted line fit")

  plt.pause(0.01)

def vis_data_line_3(dataMatrix, line_param_final, dataMatrixPart):
  plt.subplot("133")
  plt.scatter(dataMatrix[:,0], dataMatrix[:,1], s=1, c='r')
  plt.scatter(dataMatrixPart[:,0], dataMatrixPart[:,1], s=4, c='g')
  x = np.linspace(min(dataMatrix[:,0]), max(dataMatrix[:,0]), 2)
  y_final = line_param_final[0] * x + line_param_final[1]
  plt.plot(x,y_final, c='b')
  plt.axis('scaled')
  plt.title("final weighted line fit")

  plt.pause(0.01)

def vis_corner(dataMatrix, line_param):
  plt.subplot("133")
  x = np.linspace(min(dataMatrix[:,0]), max(dataMatrix[:,0]), 2)
  y_final = line_param[0] * x + line_param[1]
  plt.plot(x,y_final, c='b')
  plt.axis('scaled')
  
  plt.pause(3)

def vis(dataMatrix, line_param, line_param_weighted, line_param_final, dataMatrixPart):
  # plt.subplot("131")
  # plt.scatter(dataMatrix[:,0], dataMatrix[:,1], s=1)
  # x = np.linspace(min(dataMatrix[:,0]), max(dataMatrix[:,0]), 2)
  # y = line_param[0] * x + line_param[1] 
  # plt.plot(x,y, c='r')
  # plt.title("equal weight line fit")

  plt.subplot("132")
  plt.scatter(dataMatrix[:,0], dataMatrix[:,1], s=1)
  x = np.linspace(min(dataMatrix[:,0]), max(dataMatrix[:,0]), 2)
  y_weighted = line_param_weighted[0] * x + line_param_weighted[1]
  plt.plot(x,y_weighted, c='r')
  plt.title("weighted line fit")

  plt.subplot("133")
  plt.scatter(dataMatrix[:,0], dataMatrix[:,1], s=1, c='r')
  plt.scatter(dataMatrixPart[:,0], dataMatrixPart[:,1], s=4, c='g')
  x = np.linspace(min(dataMatrix[:,0]), max(dataMatrix[:,0]), 2)
  y_final = line_param_final[0] * x + line_param_final[1]
  plt.plot(x,y_final, c='b')
  plt.title("final weighted line fit")

  plt.show(block=True)

# main
def main():
  vis_init()
  ROW = 30
  COL = 2
  dataMatrix = np.zeros([ROW, COL]) # 
  # dataMatrix = prepareDataMatrix(dataMatrix, "line_curved")
  dataMatrix = prepareDataMatrix(dataMatrix, "corner")
  # assert(dataMatrix[3,0] == 3)

  dataHomo = np.ones([ROW,1])
  line_param, error = fit_ls_loop(dataMatrix, dataHomo)
  # line_param_ab, error, rank, singular_value = np.linalg.lstsq(dataMatrix, dataHomo)
  # line_slope = -1 * line_param_ab[0] / line_param_ab[1]
  # line_intercept = 1 / line_param_ab[1]
  # line_param = np.array([line_slope, line_intercept])
  print("error", error)
  
  line_param_final, error_final = fit_line(dataMatrix, dataHomo)
  fit_corner(dataMatrix, dataHomo, line_param_final)
  print("error_final", error_final)

  plt.show(block=True)
  print("stop")


if __name__ == "__main__":
  main()