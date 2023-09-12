import numpy as np
from sklearn.metrics import pairwise_distances

class CoresetGreedy:
    def __init__(self, all_pts):
        self.all_pts = np.array(all_pts)  # Convert input points to numpy array
        self.dset_size = len(all_pts)  # Get the total number of points
        self.min_distances = np.inf * np.ones(self.dset_size)  # Initialize minimum distances to infinity
        self.already_selected = set()  # Initialize an empty set to keep track of selected points

    def update_distances(self, new_indices):
        # Calculate distances between new indices and all points and update the minimum distances accordingly
        dists = pairwise_distances(self.all_pts[new_indices], self.all_pts, metric='euclidean')
        self.min_distances = np.minimum(self.min_distances, np.min(dists, axis=0))

    def sample(self, sample_ratio):
        # Calculate sample size based on the input ratio
        sample_size = int(self.dset_size * sample_ratio)
        new_indices = []
        for _ in range(sample_size):
            if not self.already_selected:
                # If no points have been selected, choose one at random
                new_idx = np.random.choice(self.dset_size)
            else:
                # Otherwise, select the point that maximizes the minimum distance to already selected points
                # If the point has already been selected, it sets its minimum distance to zero and finds the next point with the maximum minimum distance
                new_idx = np.argmax(self.min_distances)
                while new_idx in self.already_selected:
                    self.min_distances[new_idx] = 0
                    new_idx = np.argmax(self.min_distances)

            self.already_selected.add(new_idx)  # Add the new index to the set of selected points
            self.update_distances([new_idx])  # Update the minimum distances with the new index
            new_indices.append(new_idx)  # Add the new index to the list of new indices

        return new_indices  # Return the list of new indices

# Example usage:
# coreset = CoresetGreedy([[1,2], [3,4], [5,6], [7,8]])
# sampled_indices = coreset.sample(0.5)
# print(sampled_indices)
