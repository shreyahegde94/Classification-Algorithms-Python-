import numpy as np
from scipy.spatial.distance import euclidean
from collections import Counter
from sklearn.model_selection import KFold
from scipy import stats

def identify_cluster(training_set, labels_training_set, testing_set_i, labels_testing_set):

    distaces_list = []

    k_labels = []

    for j in range(len(training_set)):

        #dist = np.linalg.norm(testing_set_i - training_set[j])

        dist = euclidean(testing_set_i, training_set[j])

        distaces_list.append(dist)

    #print "distance_list" , distaces_list

    indexes = np.argpartition(distaces_list, k)

    shortest_k_indexes = indexes[:k]

    for x in range(len(shortest_k_indexes)):

        k_labels.append(labels_training_set[shortest_k_indexes[x]])


    count = Counter(k_labels)
    majority_label = count.most_common()[0][0]
    return majority_label



def find_kNN(training_set, labels_training_set, testing_set, labels_testing_set):

    predicted_labels = []

    for i in range(len(testing_set)):

        belong_cluster = identify_cluster(training_set, labels_training_set, testing_set[i], labels_testing_set )

        predicted_labels.append(belong_cluster)

    return predicted_labels

def performace_param(predicted_labels_ret, labels_testing_set):

    tp = 0

    tn = 0

    fp = 0

    fn = 0

    for i in range(len(predicted_labels_ret)):

        if(predicted_labels_ret[i] == labels_testing_set[i] == 1):
            tp = tp + 1
        elif(predicted_labels_ret[i] == labels_testing_set[i] == 0):
            tn = tn + 1
        elif(predicted_labels_ret[i] == 1 and labels_testing_set[i] == 0):
            fp = fp + 1
        elif(predicted_labels_ret[i] == 0 and labels_testing_set[i] == 1):
            fn = fn + 1

    return tp, tn, fp, fn


def calculate_accuracy(predicted_labels_ret, labels_testing_set):

    tp, tn, fp, fn = performace_param(predicted_labels_ret, labels_testing_set)
    if((tp + tn + fp + fn) == 0):
        return float(tp + tn)
    accuracy = float(tp + tn)/float(tp + tn + fp + fn)

    return accuracy

def calculate_precision(predicted_labels_ret, labels_testing_set):

    tp, tn, fp, fn = performace_param(predicted_labels_ret, labels_testing_set)
    if((tp + fp) == 0):

        return float(tp)

    precision = float(tp)/float(tp + fp)

    return precision


def calculate_recall(predicted_labels_ret, labels_testing_set):

    tp, tn, fp, fn = performace_param(predicted_labels_ret, labels_testing_set)
    if((tp + fn) == 0):

        return float(tp)
    recall = float(tp)/float(tp + fn)

    return recall



def calculate_fmeasure(predicted_labels_ret, labels_testing_set):

    tp, tn, fp, fn = performace_param(predicted_labels_ret, labels_testing_set)
    if((2 * tp + fp + fn) == 0):

        return float(2 * tp)
    fmeasure = float(2 * tp)/float(2 * tp + fp + fn)

    return fmeasure

print " "
k = raw_input("enter k:")
print " "
k = int(k)
training_set = np.loadtxt("project3_dataset3_train.txt")
labels_training_set = training_set[:, -1]
training_set = np.delete(training_set, -1, axis = 1)
#training_set = stats.zscore(training_set, axis=0)
training_mean = np.mean(training_set, axis = 0)
training_std = np.std(training_set, axis = 0)

training_set = (training_set - training_mean)/training_std


testing_set = np.loadtxt("project3_dataset3_test.txt")
labels_testing_set = testing_set[:,-1]
testing_set = np.delete(testing_set, -1, axis = 1)
testing_set = (testing_set - training_mean)/training_std
#testing_set = stats.zscore(testing_set, axis=0)


predicted_labels_ret = find_kNN(training_set, labels_training_set, testing_set, labels_testing_set)

accuracy = calculate_accuracy(predicted_labels_ret, labels_testing_set)

precision = calculate_precision(predicted_labels_ret, labels_testing_set)

recall = calculate_recall(predicted_labels_ret, labels_testing_set)

f1_measure = calculate_fmeasure(predicted_labels_ret, labels_testing_set)


print "Accuracy" , accuracy
print "Precision" , precision
print "Recall" , recall
print "F1_measure", f1_measure
