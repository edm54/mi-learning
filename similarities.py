import numpy as np
import math

# TODO: https://www.machinelearningplus.com/statistics/mahalanobis-distance/


# Returns min euclidean distance
def euclid(bag, prototype):
    min_dist = math.inf
    for instance in bag:
        cur_dist = euclidean_dist(instance[1:], prototype)
        if cur_dist < min_dist:
            min_dist = cur_dist
    return min_dist

def euclid_histogram(bag, prototypes):
    hist = np.zeros(len(prototypes))
    for instance in bag:
        min_dist = math.inf
        for ind, prototype in enumerate(prototypes):
            cur_dist = euclidean_dist(instance[1:], prototype)
            if cur_dist < min_dist:
                min_dist = cur_dist
                min_ind = ind
        hist[min_ind] += 1
    # normalize
    hist = hist/len(bag)
    return hist

def similarity_histogram(bag, prototypes):
    hist = np.zeros(len(prototypes))
    for instance in bag:
        max_sim = -1 * math.inf
        for ind, prototype in enumerate(prototypes):
            cur_dist = gaussian_kernel(instance[1:], prototype)
            if cur_dist > max_sim:
                max_sim = cur_dist
                max_ind = ind
        hist[max_ind] += 1

    # normalize
    hist = hist/len(bag)
    return hist

# Codework plausibility from Visual Word Ambiguity
def kernel_plausibility(bag, prototypes):
    hist = np.zeros(len(prototypes))
    for instance in bag:
        min_dist = math.inf
        min_ind = 0
        for ind, prototype in enumerate(prototypes):
            cur_dist = euclidean_dist(instance[1:], prototype)
            if cur_dist < min_dist:
                min_dist = cur_dist
                min_ind = ind

        hist[min_ind] += gaussian_codebook_kernel(instance[1:], prototypes[ind])
    return hist

def euclidean_dist(instance, prototype):
    return np.linalg.norm(np.asarray(instance) - np.asarray(prototype))

# Returns max similarity
def similarity(bag, prototype, sigma=1):
    max_sim = -1 * math.inf
    for instance in bag:
        cur_dist = gaussian_kernel(instance[1:], prototype, sigma=sigma)
        if cur_dist > max_sim:
            max_sim = cur_dist
    return max_sim

def gaussian_kernel(instance, prototype, sigma=1, coeff=1):
    dist = euclidean_dist(instance, prototype) ** 2

    coeff = len(instance) * coeff
    # # Chosen based off seed paper
    result = np.exp(-1 * dist / (coeff*(sigma ** 2)))
    return result

def gaussian_codebook_kernel(instance, prototype, sigma=1, coeff=1):
    dist = euclidean_dist(instance, prototype) ** 2
    coeff = len(instance) * coeff
    # # Chosen based off seed paper
    result = (1/(math.sqrt(2*math.pi) * sigma)) * np.exp(-1 * dist /(coeff*(sigma ** 2)))

    return result

# Codework plausibility from Visual Word Ambiguity
def kernel_codebook(bag, prototypes):
    hist = np.zeros(len(prototypes))
    for ind, prototype in enumerate(prototypes):
        for instance in bag:
            hist[ind] += gaussian_codebook_kernel(instance[1:], prototype)
    return hist

# Codework uncertainty from Visual Word Ambiguity
def kernel_codebook_uncertainty(bag, prototypes):
    hist = np.zeros(len(prototypes))
    inst_sums = np.zeros(len(bag))

    for ind,instance in enumerate(bag):
        for prototype in prototypes:
           inst_sums[ind] += gaussian_codebook_kernel(instance[1:], prototype)

    # Find dist from this instance to all other clusters
    # Divide instance sum by dist
    for ind, prototype in enumerate(prototypes):
        for i,instance in enumerate(bag):
            hist[ind] += gaussian_codebook_kernel(np.asarray(instance[1:]), prototype)/inst_sums[i]

    return hist

def yards_sim(bag,prototype):
    sum = 0
    for inst in bag:
        sum += gaussian_kernel(np.asarray(inst[1:]), np.asarray(prototype))
    return sum

def mahalanobis(instance, prototype):
    cov = np.cov(prototypes.T)


