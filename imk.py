import random
import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import EDM54.miHelperMethods as utilities
import EDM54.similarities as similarities
from sklearn.metrics.pairwise import euclidean_distances
#import skfuzzy as fuzz
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics.pairwise import rbf_kernel
num_clusters = 0
cluster_centers = None
class IMK:

    def learn(self, bags, clusters=8,c=1):
        global num_clusters
        global cluster_centers
        num_clusters = clusters
        instances = utilities.get_all_intances(bags)

        # Cluster instances  --> k classes
        cluster_centers = self.cluster_instances(instances, num_clusters=clusters)

        V, labels = self.map_all_imk(bags, cluster_centers)

        # Use SVM to classify the vector V
        # C is penalty parameter
        # Custom Kernel
        classifier = SVC(kernel=imk_kernel,C=c, gamma = 'auto')
        classifier.fit(V, labels)

        return cluster_centers, classifier

    def map_all_imk(self, bags, prototypes):
        mapped_vectors = []
        labels = []
        for b in bags:
            v = self.map_to_vector_imk(b[1], prototypes)
            labels.append(b[-1])
            mapped_vectors.append(v)
        return mapped_vectors, labels

    # Map a single bag to a vector
    def map_to_vector_imk(self, bag, prototypes):

        # v = [[None]*len(bag[1])] * len(prototypes)
        v = []
        # Vector will be length of an instance in the bag

        for ind, prototype in enumerate(prototypes):
            min_dist = math.inf
            min_instance = [None]

            # Each 'index' of the vector is the instance that best matches that class
            # Concatenate vectors together
            for index, instance in enumerate(bag):
                current_dist = similarities.euclidean_dist(instance[1:], prototype)
                if current_dist < min_dist:
                    min_dist = current_dist
                    min_instance = instance
            v.extend(min_instance[1:])
        return v

    def cluster_instances(self, instances, num_clusters=8):
        km = KMeans(n_clusters=num_clusters).fit(instances)
        return km.cluster_centers_

# Custom kernel for IMK
def imk_kernel(x,y):
    # Y is col, X is row
    # Create matrix with X row, Y col
    out_matrix = [[0 for i in range(len(y))] for j in range(len(x))]

    # Find each pair of vectors in  x,y
    for ind, vector in enumerate(x):
        for ind_y,vector_y in enumerate(y):

            # Find subvector length based on the number of clusters
            vector_l = len(vector)
            sub_vector_l = vector_l/num_clusters
            kernel_sum = 0

            # For each subvector, find guassian kernel with coeff of 2
            for i in range(num_clusters):
                start = int(0 + i*sub_vector_l)
                end = int((i+1) * sub_vector_l)
                sub_v = vector[start:end]
                sub_w = vector_y[start:end]
                kernel_sum += similarities.gaussian_kernel(sub_v, sub_w, coeff=2, sigma=.5)
            out_matrix[ind][ind_y] = kernel_sum

    return out_matrix


def main():

    print('run with db MI')

if __name__ == "__main__":
    main()