import os
from sklearn.model_selection import KFold
from torch import optim, nn, utils, Tensor, rand
import torch

from torchvision.transforms import ToTensor
from torchvision import datasets
import torchvision.transforms as transforms
import lightning as L
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader

from sklearn.mixture import GaussianMixture

from util import dist
import pandas as pd
import numpy as np
import logging
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import pdb

from scipy.signal import savgol_filter

def smoothen (values,
              window_size = 11,  # Choose an appropriate window size
              poly_order = 3    # Choose an appropriate polynomial order
             ):
  return savgol_filter(values, window_size, poly_order)


def calculate_mbr(trajectories, is_plot=True) :
  # Calculate the convex hull for each trajectory
  convex_hulls = []
  for trajectory in trajectories:
      points = np.array(trajectory)
      hull = ConvexHull(points)
      convex_hulls.append(hull)
  combined_hull = ConvexHull(np.vstack([hull.points for hull in convex_hulls]))
  if is_plot:
    # Plot the combined convex hull (MBR)
    plt.plot(*combined_hull.points[combined_hull.vertices, :].T, 'r--', lw=2)
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.scatter(*np.array(trajectories).T, s=10)  # Scatter plot of your trajectories
    plt.show()

  return combined_hull


# Check if the new trajectory is inside the convex hull (MBR)
def is_inside(traj, hull):
  def is_inside_hull(point, hull):
    return all(
        np.dot(eq[:-1], point) + eq[-1] <= 0
        for eq in hull.equations
    )
  return all(is_inside_hull(point, hull) for point in traj)

def smoothen (values,
              window_size = 11,  # Choose an appropriate window size
              poly_order = 3    # Choose an appropriate polynomial order
             ):
  return savgol_filter(values, window_size, poly_order)

def run_wildgen(df_train, data):
    real = df_train[['label', 'location.lat', 'location.long']].groupby('label')
    group_arrays = [group[['location.lat', 'location.long']].to_numpy() for _, group in real]
    train_array = np.array(group_arrays)
    mbr_hull = calculate_mbr(train_array)
    smooth_trajectories = [np.column_stack((smoothen(traj[0]), smoothen(traj[1]))) for traj in data.T]
    filtered_trajectories = [trajectory for trajectory in smooth_trajectories  if is_inside(trajectory, mbr_hull)]
    indexes = range(len(filtered_trajectories))
    random = np.random.choice(indexes, size=60, replace=False)
    return [filtered_trajectories[index] for index in random] #filtered_trajectories[0:60]
