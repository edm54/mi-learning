import mldata as mldata
import random
import copy
import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier
import EDM54.miHelperMethods as utilities
import data_parser as dp
from EDM54.imk import IMK
from EDM54.dbow import DBOW
from EDM54.hbow import HBOW
from EDM54.yards import YARDS
from EDM54.simpleMi import SMI
import multiprocessing
import EDM54.similarities as similarities
import pandas as pd

std_list = ['imk', 'dbow', 'hbow', 'yards', 'smi']

class embeddedSpace:

    def ES_MI(self, path, cross_validation, learning_algorithm, clusters=8, distance_metric='euclidean',
                          cluster_scheme='one', C=1):
        if type(clusters) is list and len(clusters) == 1:
            clusters = clusters[0]
        # Standardize data
        if learning_algorithm in std_list:
            examples = dp.parse_data_file(path)
            attribute_map = utilities.classify_attributes(path)
            std_examples = utilities.standardize_examples(examples, attribute_map)
            bags = dp.make_bags(std_examples)
        else:
            bags = dp.getBaggedData(path)

        if cross_validation == 0:
             self.cross_validation(bags, learning_algorithm, folds=10, iterations=2,
                                   num_clusters=clusters, distance_metric = distance_metric, c=C)

        elif learning_algorithm is 'dbow':
            dbow = DBOW()
            dbow.distance_bow(bags, clusters=clusters, distance_metric=distance_metric)

        elif learning_algorithm is 'yards':
            yards = YARDS()
            yards.learn(bags)

        elif learning_algorithm is 'noncluster':
            self.nonclustering_BOW(bags)

        elif learning_algorithm is 'imk':
            imk = IMK()
            imk.learn(bags, clusters=clusters)

        elif learning_algorithm is 'hbow':
            hbow = HBOW()
            hbow.histogram_bow(bags, clusters=clusters, distance_metric=distance_metric)
        else:
            raise Exception('Error: Algorithm not recognized')

    # use for fox or elephant
    def ES_MI2(self, animal, cross_validation, learning_algorithm, clusters=8, distance_metric='euclidean',
                          cluster_scheme='one', C=1):
        if type(clusters) is list and len(clusters) == 1:
            clusters = clusters[0]

        print(animal)
        # Standardize data
        if learning_algorithm in std_list:
            if animal is 'elephant':
                _, labels, data, l2, indexes = dp.loadelephant()
            else:
                _, labels, data, l2, indexes = dp.loadfox()

            data = utilities.standardize_everything(data)

            bags = dp.parse_svm(data, labels, indexes)


        else:
            if animal is 'elephant':
                _, labels, data, l2, indexes = dp.loadelephant()
            else:
                _, labels, data, l2, indexes = dp.loadfox()
            #bags = dp.getBaggedData(path)

            bags = dp.parse_svm(data, labels, indexes)

        if cross_validation == 0:
             self.cross_validation(bags, learning_algorithm, folds=10,iterations=3,
                                   num_clusters=clusters, distance_metric = distance_metric, c=C)

        elif learning_algorithm is 'dbow':
            dbow = DBOW()
            dbow.distance_bow(bags, clusters=clusters, distance_metric=distance_metric)

        elif learning_algorithm is 'yards':
            yards = YARDS()
            yards.learn(bags)

        elif learning_algorithm is 'noncluster':
            self.nonclustering_BOW(bags)

        elif learning_algorithm is 'imk':
            imk = IMK()
            imk.learn(bags, clusters=clusters)

        elif learning_algorithm is 'hbow':
            hbow = HBOW()
            hbow.histogram_bow(bags, clusters=clusters, distance_metric=distance_metric)
        else:
            raise Exception('Error: Algorithm not recognized')



    def cluster_instances(self, instances, num_clusters = 8):
        km = KMeans(n_clusters=num_clusters).fit(instances)
        return km.cluster_centers_

    def cross_validation(self, bags, learning_algorithm, folds=5, iterations=1, num_clusters=8,
                         distance_metric='euclidean',c=1):


        bins = folds
        class_one = []
        class_two = []
        accuracies = []
        precisions = []
        recalls = []
        # find out how many in each class
        for i in range(0, len(bags)):
            if bags[i][-1]:
                class_one.append(i)
            elif not bags[i][-1]:
                class_two.append(i)

        # set the random seed
        #np.random.seed(12345)
        for iter in range(0,iterations):
            # randomly shuffle the two
            np.random.shuffle(class_one)
            np.random.shuffle(class_two)

            for a in range(0, bins):
                testing_class_one = class_one[len(class_one) // bins * a:len(class_one) // bins * (a + 1)]
                testing_class_two = class_two[len(class_two) // bins * a:len(class_two) // bins * (a + 1)]

                training_class_one = [a for a in class_one if a not in testing_class_one]
                training_class_two = [a for a in class_two if a not in testing_class_two]

                all_class_labels_test = \
                    self.select_subset(bags, testing_class_one, testing_class_two)

                all_class_labels_train = \
                    self.select_subset(bags, training_class_one, training_class_two)

                #for a in all_class_labels_train:
                    #print(a[-1])

                if learning_algorithm is 'dbow':
                    dbow = DBOW()
                    cluster_centers, classifier = dbow.distance_bow(all_class_labels_train, num_clusters,
                                                                    distance_metric=distance_metric,c=c)
                    test_vector, test_labels = dbow.map_all(all_class_labels_test, cluster_centers,
                                                            num_clusters=num_clusters,
                                                            distance_metric=distance_metric)
                    results = classifier.predict(test_vector)

                elif learning_algorithm is 'imk':
                    imk = IMK()
                    cluster_centers, classifier = imk.learn(all_class_labels_train, clusters=num_clusters, c=c)
                    test_vector, test_labels = imk.map_all_imk(all_class_labels_test, cluster_centers)
                    results = classifier.predict(test_vector)

                elif learning_algorithm is 'hbow':
                    hbow = HBOW()
                    cluster_centers, classifier = hbow.histogram_bow(all_class_labels_train, num_clusters,
                                                                     distance_metric=distance_metric, C=c)
                    test_vector, test_labels = hbow.map_all(all_class_labels_test, cluster_centers,
                                                            num_clusters=num_clusters,
                                                            distance_metric=distance_metric)
                    results = classifier.predict(test_vector)

                elif learning_algorithm is 'yards':
                    yards = YARDS()
                    cluster_centers, classifier = yards.learn(all_class_labels_train, c=c)
                    test_vector, test_labels = yards.map_all(all_class_labels_test,cluster_centers)
                    results = classifier.predict(test_vector)

                elif learning_algorithm is 'smi':
                    smi = SMI()
                    classifier = smi.learn(all_class_labels_train, c=c)
                    test_vector, test_labels = smi.map_all(all_class_labels_test)
                    results = classifier.predict(test_vector)

                else:
                    raise Exception('Not recognized')

                true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0

                for ind, actual_labels in enumerate(test_labels):
                    res = int(results[ind])
                    if actual_labels > 0:
                        actual_labels = True
                    else:
                        actual_labels = False
                       # actual_labels = int(actual_labels)
                    #if type(result) is not int:
                    if res > 0:
                        res = True
                    else:
                        res = False
                        #results[ind] = int(results[ind])

                    if actual_labels and res:  # true positives
                        true_positives = true_positives + 1

                    if actual_labels and not res:  # false negative
                        false_negatives = false_negatives + 1

                    if not actual_labels and not res:  # true negative
                        true_negatives = true_negatives + 1

                    if not actual_labels and res:  # false postitives
                        false_positives = false_positives + 1

                accuracy = (true_positives + true_negatives) / len(all_class_labels_test)
                if true_positives + false_positives is 0:
                    precision = 0
                else:
                    precision = true_positives / (true_positives + false_positives)
                recall = true_positives / (true_positives + false_negatives)

                print('accuracy: ', accuracy)

                #print(accuracy)

                accuracies.append(accuracy)
                precisions.append(precision)
                recalls.append(recall)
                #print(accuracies)
                #print(np.average(accuracies))

        final_accuracy = np.sum(accuracies) / (bins * iterations)
        final_precision = np.sum(precisions) / (bins * iterations)
        final_recall = np.sum(recalls) / (bins * iterations)

        #print(accuracies)
        #print(precisions)
        #print(recalls)
        ci = 1.96  * np.std(accuracies)/math.sqrt(len(accuracies))
        ci_t = ci + final_accuracy
        ci_b = final_accuracy - ci
        print(c, distance_metric, learning_algorithm, num_clusters)
        print("Accuracy: %.3f" % final_accuracy, " %.3f" % np.std(accuracies))
        print("Precision: %.3f" % final_precision, " %.3f" % np.std(precisions))
        print("Recall: %.3f" % final_recall, " %.3f" % np.std(recalls))
        print("CI: %.4f " %ci, " %.4f " % ci_b , "%.4f" % ci_t)
        print()


    def select_subset(self, bags, part_class_one, part_class_two):
        """
        Categorize each example based on class label
        :param examples: example datasets
        :param part_class_one: examples with positive class label
        :param part_class_two: examples with negative class label
        :return:
        """

        new_bags = []

        for i in range(0, len(part_class_one)):
            new_bags.append(bags[part_class_one[i]])
        for i in range(0, len(part_class_two)):
            new_bags.append(bags[part_class_two[i]])

        return new_bags


def tune_c(c):
    es = embeddedSpace()
    es.ES_MI2('elephant', 0, 'dbow', distance_metric='similarity', clusters=512, C=c)


def pooled_C_tuning():
    c_c = [.5, 1,10,100]
    pool = multiprocessing.Pool()
    pool.map(tune_c, [c for c in c_c])
    pool.close()



def main():
    # _,labels,data,l2,indexes = dp.loadelephant()
    # bags = dp.parse_svm(data, labels, indexes)
    es = embeddedSpace()
    es.ES_MI('musk1', 0, 'dbow', distance_metric='euclidean', C=10, clusters=128)




if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()