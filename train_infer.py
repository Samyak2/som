import numpy as np
import torch
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
    data.append(colors[i, :])
data_t = torch.Tensor(data)

plt.ion()

def draw_som(som):
    plt.clf()

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
    som(data_t, iter_no)

mapped = som.map_vects(torch.Tensor(colors))

def infer(som, color_rgb):
    required = som.get_bmu_loc(torch.Tensor(color_rgb))

    dists = som.pdist(torch.stack([required for i in range(len(mapped))]), torch.stack(mapped))

    min_dist, min_dist_index = torch.min(dists, 0)

    color_name = color_names[min_dist_index]

    return float(min_dist), required, color_name

def infer_plot(som, color_rgb):
    min_dist, bmu_loc, color_name = infer(som, color_rgb)

    draw_som(som)

    plt.text(bmu_loc[1], bmu_loc[0], "INPUT", ha="center", va="center",
             bbox=dict(facecolor='white', alpha=0.5, lw=0))

    return min_dist, bmu_loc, color_name

print(infer(som, [0.0, 0.0, 0.0]))
