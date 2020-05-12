import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import EDM54.miHelperMethods as utilities
import EDM54.similarities as similarities
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import chi2_kernel


class YARDS:

    def learn(self, bags, c=10, sigma=1):
        # TODO: Try with just positive instances?
        #instances = utilities.get_all_intances(bags)
        instances = utilities.get_all_positive_intances(bags)
        # Don't cluster, uses all instances

        # Mapping Function: Given X and Vocab V, return vector V
        V, labels = self.map_all(bags, instances)

        # Use SVM with Chi Square as kernel
        classifier = SVC(kernel='rbf', gamma= 'auto', C=c)
        classifier.fit(V, labels)
        return instances, classifier


    def map_all(self, bags, prototypes):
        mapped_vectors = []
        labels = []
        for b in bags:
            v = self.map_to_vector(b[1], prototypes)
            labels.append(b[-1])
            mapped_vectors.append(v)
        return mapped_vectors, labels

    def map_to_vector(self, bag, prototypes):
        v = [0] * len(prototypes)
        for ind, prototype in enumerate(prototypes):
            similarity = similarities.yards_sim(bag, prototype)
            v[ind] = similarity

        return v

    def cluster_instances(self, instances, num_clusters=8):
        km = KMeans(n_clusters=num_clusters).fit(instances)
        return km.cluster_centers_
