import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import EDM54.miHelperMethods as utilities
import EDM54.similarities as similarities
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import chi2_kernel


class SMI:

    def learn(self, bags, c=1, kernel='rbf'):
        instances = utilities.get_all_intances(bags)
        #instances = utilities.get_all_positive_intances(bags)
        # Don't cluster, uses all instances

        # Mapping Function: Given X and Vocab V, return vector V
        V, labels = self.map_all(bags)


        # Use SVM with rbf as kernel
        classifier = SVC(kernel= 'rbf', gamma= 'auto', C=10)

        classifier.fit(V, labels)
        return classifier


    def map_all(self, bags):
        mapped_vectors = []
        labels = []
        for b in bags:
            v = self.map_to_vector_min(b[1])
            labels.append(b[-1])
            mapped_vectors.append(v)
        return mapped_vectors, labels

    # avg vector
    def map_to_vector(self, bag):
        v = [0] * len(bag[0][1:])
        for instance in bag:
            for ind,i in enumerate(instance[1:]):
                v[ind] += i
        v = [i/len(bag) for i in v]
        #print(v)

        return v


    # min max vector
    def map_to_vector_min(self, bag):
        v = [0] * len(bag[0][1:]) * 2
        for instance in bag:
            for ind,i in enumerate(instance[1:]):
                v[ind] = min(i, v[ind])
                v[ind+len(bag[0][1:])] = max(i,v[ind+len(bag[0][1:])])
        #print(v)

        return v


    def map_to_vector_mma(self, bag):
        v = [0] * len(bag[0][1:]) * 3
        for instance in bag:
            for ind,i in enumerate(instance[1:]):
                v[ind] = min(i, v[ind])
                v[ind+len(bag[0][1:])] = max(i, v[ind+len(bag[0][1:])])
                v[ind + 2*len(bag[0][1:])] += i

        #print(v)

        for i,ins in enumerate(v[2*len(bag[0][1:]):]):
            v[i] = v[i]/len(bag)


        return v