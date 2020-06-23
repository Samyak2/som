"""
Much of the code is modified from:
- https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/
"""

import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

from som import SOM

m = 20
n = 30

#Training inputs for RGBcolors
colors = np.array([[0., 0., 0.],
                   [0., 0., 1.],
                   [0., 0., 0.5],
                   [0.125, 0.529, 1.0],
                   [0.33, 0.4, 0.67],
                   [0.6, 0.5, 1.0],
                   [0., 1., 0.],
                   [1., 0., 0.],
                   [0., 1., 1.],
                   [1., 0., 1.],
                   [1., 1., 0.],
                   [1., 1., 1.],
                   [.33, .33, .33],
                   [.5, .5, .5],
                   [.66, .66, .66]])
color_names = \
    ['black', 'blue', 'darkblue', 'skyblue',
     'greyblue', 'lilac', 'green', 'red',
     'cyan', 'violet', 'yellow', 'white',
     'darkgrey', 'mediumgrey', 'lightgrey']

data = list()
for i in range(colors.shape[0]):
    data.append(torch.FloatTensor(colors[i, :]))

plt.ion()

def draw_som(som):
    #Store a centroid grid for easy retrieval later on
    centroid_grid = [[] for i in range(m)]
    weights = som.get_weights()
    locations = som.get_locations()
    for i, loc in enumerate(locations):
        centroid_grid[loc[0]].append(weights[i].numpy())

    #Get output grid
    image_grid = centroid_grid

    #Map colours to their closest neurons
    mapped = som.map_vects(torch.Tensor(colors))

    #Plot
    plt.imshow(image_grid)
    plt.title('Color SOM')
    for i, map_ in enumerate(mapped):
        plt.text(map_[1], map_[0], color_names[i], ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.5, lw=0))

#Train a 20x30 SOM with 100 iterations
n_iter = 100
som = SOM(m, n, 3, n_iter)
for iter_no in range(n_iter):
    #Train with each vector one by one
    for data_i in data:
        som(data_i, iter_no)

    if iter_no % 5 == 0:
        draw_som(som)
        plt.pause(0.0001)
        plt.clf()
        print(f"Trained {iter_no} iterations")
plt.ioff()
draw_som(som)
plt.show()
