
import matplotlib.pyplot as plt

class KMeans:
  def __init__(self,X,num_clusters,max_iters):
    self.num_clusters = num_clusters
    self.max_iters = max_iters
    self.num_examples,self.num_features = X.shape #rows = number of examples
                                                  #columns = number of features

  def initialize_centroids(self,X):
    centroids = np.zeros((self.num_clusters,self.num_features))
    for cluster_id in range(self.num_clusters):
      centroid = X[np.random.choice(range(self.num_examples))]
      centroids[cluster_id] = centroid
    return centroids

  def create_clusters(self,centroids,X):
    clusters = [[] for _ in range(self.num_clusters)]
    for point_id,point in enumerate(X):
      closest_centroid = np.argmin(np.sqrt((np.sum((point-centroids)**2,axis=1))))
      clusters[closest_centroid].append(point_id)
    return clusters

  def calculate_new_centroids(self,clusters,X):
    centroids = np.zeros((self.num_clusters,self.num_features))
    for cluster_id,cluster in enumerate(clusters):
      if(len(cluster) == 0):
        centroids[cluster_id] = X[np.random.choice(self.num_examples)]
      else:
        new_centroid = np.mean(X[cluster],axis=0)
        centroids[cluster_id] = new_centroid
    return centroids

  def train(self,X):
    centroids = self.initialize_centroids(X)
    for _ in range(self.max_iters):
      clusters = self.create_clusters(centroids,X)
      previous_centroids = centroids
      centroids = self.calculate_new_centroids(clusters,X)
      diff = centroids - previous_centroids
      if not diff.any():
        break
    return clusters,centroids

  def plot(self, X, clusters, centroids,labels):
      colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown']

      for cluster_id, cluster in enumerate(clusters):
          points = X[cluster]  # Get all points for this cluster
          plt.scatter(points[:, 0], points[:, 1], s=50, c=colors[cluster_id % len(colors)], label=f'Cluster {cluster_id + 1}')

      # Plot centroids
      plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black', marker='X', label='Centroids')

      plt.title('Clusters of Customers')
      plt.xlabel(labels[0])
      plt.ylabel(labels[1])
      plt.legend()
      plt.show()
