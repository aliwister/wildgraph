from scipy.spatial.distance import directed_hausdorff
import numpy as np
from scipy.spatial.distance import euclidean
from tslearn.metrics import dtw as ts_dtw
import torch

class Hausdorff():
    #def __init__(self) -> None:
        #df = pd.read_csv('trajs.csv')
        #real = df[['label', 'location.lat', 'location.long']].groupby('label')
        #real.head()
        #group_arrays = [group[['location.lat', 'location.long']].to_numpy() for _, group in real]
        #self.real_array = np.array(group_arrays)
         
    def update(self, real, generated):
        distances = []
        touched = np.zeros(len(real))
        for traj1 in generated:
            path = 100000
            path_idx = -1
            skip = False
            for i, traj2 in enumerate(real):
                #pdb.set_trace()
                try:
                    if(len(traj1) != len(traj2)):
                        #print ("Skipping Short Traj")
                        skip = True
                        break
                except TypeError:
                    print("Type Error", traj1, generated)
                    return 0,0,0
                distance1 = directed_hausdorff(traj1, traj2)[0]
                distance2 = directed_hausdorff(traj2, traj1)[0]
                distance = np.max([distance1, distance2])
                if distance < path:
                    path = distance
                    path_idx = i
            touched[path_idx] = 1
            distances.append(path)

        if (len(distances) < 1):
            return -1, -1, -1
        min = np.min(distances)
        max = np.max(distances)
        mean = np.mean(distances)
        return min, max, mean, sum(touched)
        #return -1, -1, -1

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def convert_to_numpy(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy()
    else:
        return obj

class DTW():

    
    def update(self, real, generated):
        distances = []
        touched = np.zeros(len(real))
        for traj1 in generated:
            path = 100000
            skip = False
            for i, traj2 in enumerate(real):
                #pdb.set_trace()
                try:
                    if(len(traj1) != len(traj2)):
                        #print ("Skipping Short Traj")
                        skip = True
                        break
                except TypeError:
                    print("Type Error", traj1, generated)
                    return 0,0,0
                
                distance = ts_dtw(traj1, convert_to_numpy(traj2))

                if distance < path:
                    path = distance
                    path_idx = i
            touched[path_idx] = 1
            distances.append(path)

        #if (not skip):
        if (len(distances) < 1):
            return -1, -1, -1
        min = np.min(distances)
        max = np.max(distances)
        mean = np.mean(distances)
        return min, max, mean, sum(touched)
        #return -1, -1, -1    

class FDE():
    def update(self, real, generated):
        distances = []
        touched = np.zeros(len(real))
        for traj1 in generated:
            path = 100000
            skip = False
            for i, traj2 in enumerate(real):
                #pdb.set_trace()
                try:
                    if(len(traj1) != len(traj2)):
                        #print ("Skipping Short Traj")
                        skip = True
                        break
                except TypeError:
                    print("Type Error", traj1, generated)
                    return 0,0,0
                
                distance = self.distance(traj1, traj2)

                if distance < path:
                    path = distance
                    path_idx = i
            touched[path_idx] = 1
            distances.append(path)

        if (len(distances) < 1):
            return -1, -1, -1
        min = np.min(distances)
        max = np.max(distances)
        mean = np.mean(distances)
        return min, max, mean, sum(touched)
        #return -1, -1, -1    
    
    def distance(self, s1, s2):
        n = len(s1)
        m = len(s2)
        
        s1_last = s1[n-1]
        s2_last = s2[n-1]
        return euclidean_distance(s1_last, s2_last)

from sklearn.cluster import KMeans
def r_calc(X,Y,K):
  cluster_A = {}
  cluster_B = {}
  for i in range(K):
    cluster_A[i] = np.sum(X == i)
    cluster_B[i] = np.sum(Y == i)

  A = np.array([cluster_A[cluster] for cluster in sorted(cluster_A.keys())])
  B = np.array([cluster_B[cluster] for cluster in sorted(cluster_B.keys())])
  return np.corrcoef(A,B)

def chi_calc(X,Y, K):
  cluster_A = {}
  cluster_B = {}
  for i in range(K):
    cluster_A[i] = np.sum(X == i)
    cluster_B[i] = np.sum(Y == i)

  A = np.array([cluster_A[cluster] for cluster in sorted(cluster_A.keys())])
  B = np.array([cluster_B[cluster] for cluster in sorted(cluster_B.keys())])
  contingency_table = np.array([A, B])

  # Calculate expected frequencies assuming independence
  expected = np.outer(contingency_table.sum(axis=1), contingency_table.sum(axis=0)) / contingency_table.sum()

  # Calculate chi-square statistic
  chi_square = np.sum((contingency_table - expected)**2 / expected)
  return chi_square

import pdb
class Corr:
    def update(self, real, generated, k):
        print("Generated Length", len(generated))
        kmeans = KMeans(n_clusters=k)
        points_data = np.array(real).reshape(-1,2)
        kmeans.fit(points_data)

        # Get cluster assignments for each point
        cluster_assignments = kmeans.labels_
        A = kmeans.predict(points_data)
        #pdb.set_trace()
        B = kmeans.predict(np.array(generated).reshape(-1,2).astype(float))

        r_ = r_calc(A, B, k)
        chi_ = chi_calc(A, B, k)
        #pdb.set_trace()
        return r_[0][1], chi_
        #return -1, -1, -1    