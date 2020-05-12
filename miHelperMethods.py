import data_parser as dp
import numpy as np

def get_all_intances(bags):
    instances = []
    for bag in bags:
        for inst in bag[1]:
            instances.append(inst[1:][:])
    return instances

def get_all_positive_intances(bags):
    instances = []
    for bag in bags:
        if bag[-1]:
            for inst in bag[1]:
                instances.append(inst[1:][:])
    return instances



def get_all_intances_refrences(bags):
    instances = []
    for bag in bags:
        for inst in bag[1]:
            l_to_i = inst[0:]
            l_to_i.insert(0, bag[0])
            instances.append(l_to_i)

    return instances

def classify_attributes(path):
    """
    Classifies examples on nominal vs continuous (using MLData data structures)
    :param examples: dataset
    :return: Type of the attribute
    """
    attribute_type = {}
    ind = 1
    examples = dp.parse_data_file(path)
    for ex in examples.schema[1:-1]:
        if ex.type == 'NOMINAL':
            attribute_type[ind] = [ex.type, ex.values]
        # Does not store all values for a continuous attribute
        else:
            attribute_type[ind] = [ex.type]
        ind += 1

    return attribute_type

# Should standardize with kernels or distance
# STD of the kernel is 1ish should be trained, is diff from the sample
def standardize_examples(examples, attribute_map):
    """
    Standardize examples
    :param examples: dataset
    :param attribute_map: maps attribute to type of attribute & values (for nominal attributes)
    :return: standardized examples
    """

    # Create list of only continuous data
    continuous_att = [i-1 for i in attribute_map if attribute_map[i][0] == 'CONTINUOUS']
    for i in continuous_att:
        column = []
        sum = 0
        for ex in examples:
            column.append(ex[i+1])
            sum += ex[i+1]
        mean = np.mean(column)
        std = np.std(column)
        for ex in examples:
            ex[i+1] = (ex[i+1] - mean) / std

    return examples

def put_in_bags(examples):
    bag_dict = {}
    for ex in examples:

        if ex[0] in bag_dict:
            bag_dict[ex[0]][0].append(ex[1:-1])
        else:
            bag_dict[ex[0]] = ([ex[1:-1]], ex[-1])
    ex_list = [None] * len(bag_dict)
    for ind, ex in enumerate(bag_dict):
        ex_list[ind] = (ex, bag_dict[ex][0], bag_dict[ex][1])
        #print(ex_list[ind])

    return ex_list


# Should standardize with kernels or distance
# STD of the kernel is 1ish should be trained, is diff from the sample
def standardize_everything(examples):
    """
    Standardize examples
    :param examples: dataset
    :return: standardized examples
    """
    data = examples[:]
    # Create list of only continuous data
    for i in range(0,len(data[0])):
        column = []
        sum = 0
        for ex in examples:
            column.append(ex[i])
            sum += ex[i]
        mean = np.mean(column)
        std = np.std(column)
        for ex in examples:
            if std>1e-5:
                ex[i] = (ex[i] - mean) / std
            else:
                ex[i] = ex[i]-mean

    return examples



