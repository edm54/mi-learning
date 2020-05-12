import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import EDM54.miHelperMethods as utilities
import EDM54.similarities as similarities
from sklearn.ensemble import  AdaBoostClassifier


class DBOW:

    def distance_bow(self, bags, clusters=8, distance_metric='euclidean', c=1):
        instances = utilities.get_all_intances(bags)

        # TODO: Positive only!

        if type(clusters) is int or len(clusters) == 1:
            # Cluster instances  --> k classes
            # Vocab is made of k classes  described by theta
            # Prototpye Pj is vector with average of the instances
            #cluster_centers = self.cluster_instances(instances, num_clusters=clusters)

            cluster_centers = utilities.get_all_positive_intances(bags)

            # Mapping Function: Given X and Vocab V, return vector V
            # Where Vj = min(distance j to xi) for all xi in X
            # Low Vj means good matching to X
            V, labels = self.map_all(bags, cluster_centers, distance_metric)
        else:
            cluster_centers, V,labels = self.get_all_cc(instances,bags,distance_metric, clusters)


        # Use standard classifier like Adaboost to classify the vector V
        #classifier = AdaBoostClassifier(n_estimators=5000)
        # classifier.fit(V, labels)
        #‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
        if distance_metric is 'similarity' or 'sim' in distance_metric:
            print('sss')
            classifier = SVC(gamma='scale', kernel='rbf', C=c)
        else:
            print('euc')
            classifier = SVC(gamma='auto', kernel='rbf', C=c)
        # Scale for similarity
        # Auto for euclidean

        classifier.fit(V, labels)

        return cluster_centers, classifier

    def get_all_cc(self, instances, bags, distance_metric, clusters):
        vector = [[0] * len(bags)]
        labels = [0] * len(bags)
        cc = [[0] * sum(clusters)]

        for ind,c in enumerate(clusters):
            cluster_centers = self.cluster_instances(instances, num_clusters=c).tolist()
            v, l = self.map_all(bags, cluster_centers, distance_metric)

            if ind == 0:
                labels = l[:]
                vector = v[:]
                cc = cluster_centers[:]

            else:
                for i in range(0,len(bags)):
                    vector[i].extend(v[i])
                for i in range(0,len(cluster_centers)):
                    cc.append(cluster_centers[i])
       #print(len(vector[0]))
        return cc, vector, labels

    def map_all_cc(self,bags,cluster_centers, clusters, distance_metric):
        vector = [[0] * len(bags)]
        labels = [0] * len(bags)
        idx = 0
        for ind, c in enumerate(clusters):
            cc = cluster_centers[idx:idx + c]
            v, l = self.map_all(bags, cc, distance_metric)
            if ind == 0:
                labels = l[:]
                vector = v[:]
            else:
                for i in range(0, len(bags)):
                    vector[i].extend(v[i])
            idx += c
        return vector, labels

    def map_all(self, bags, prototypes, distance_metric='euclidean', num_clusters=8):
        if type(num_clusters) is int or len(num_clusters) == 1:
            labels = []
            mapped_vectors = []
            for b in bags:
                v = self.map_to_vector(b[1], prototypes, distance_metric)
                labels.append(b[-1])
                mapped_vectors.append(v)
            return mapped_vectors, labels
        else:
            return self.map_all_cc(bags,prototypes, num_clusters, distance_metric)

    def map_to_vector(self, bag, prototypes, distance_metric='euclidean'):
        v = [0] * len(prototypes)
        for ind, prototype in enumerate(prototypes):
            # TODO: incorporate different distance functions
            if distance_metric is 'euclidean' or 'euc' in distance_metric:
                similarity = similarities.euclid(bag, prototype)
            elif distance_metric is 'similarity' or 'sim' in distance_metric:
                similarity = similarities.similarity(bag, prototype, sigma=1)
            else:
                print('Unrecognized Distance Metric: Using euclidean')
                similarity = similarities.euclid(bag, prototype)

            v[ind] = similarity
        return v

    def cluster_instances(self, instances, num_clusters=8):
        km = KMeans(n_clusters=num_clusters).fit(instances)
        return km.cluster_centers_

