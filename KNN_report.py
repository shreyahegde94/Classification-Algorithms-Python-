import numpy as np
from scipy.spatial.distance import euclidean
from collections import Counter
from sklearn.model_selection import KFold
from scipy import stats

def identify_cluster(training_set, labels_training_set, testing_set_i, labels_testing_set):

    distaces_list = []

    k_labels = []

    for j in range(len(training_set)):

        dist = np.linalg.norm(testing_set_i - training_set[j])

        #dist = euclidean(testing_set_i, training_set[j])

        distaces_list.append(dist)

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

result_Data = []
with open('project3_dataset2.txt','r') as f:
    for line in f:
        result_Data.append(line.strip().split('\t'))


rows = len(result_Data)
cols = len(result_Data[0])

for i in range(rows):
    for j in range(cols):
        if not(result_Data[i][j].isalpha()):
            result_Data[i][j] = float(result_Data[i][j])

result_Data = np.array(result_Data, dtype = object)


str_list = []
for i in range(cols):
    r = result_Data[0][i]
    if(isinstance(r, str)):

        str_list.append(i)

for x in str_list:
    temp = result_Data[:,x]
    d = {ni: indi for indi, ni in enumerate(set(temp))}
    numbers = [d[ni] for ni in temp]
    result_Data[:,x] = numbers


data_from_file = result_Data

data_from_file = np.array(data_from_file, dtype = float)

#normalise the matrix

# data_normalized = (data_from_file - np.min(data_from_file, axis = 0))/ np.ptp(data_from_file, axis = 0)
# data_from_file = data_normalized
data_from_file = stats.zscore(data_from_file, axis = 0)


#data_from_file = np.genfromtxt("project3_dataset1.txt", delimiter='\t')
#
data_from_file = np.delete(data_from_file, -1, axis = 1)
#
#data_from_file = data_from_file[:,~np.all(np.isnan(data_from_file), axis=0)]

#print data_from_file[0][4]

labels = np.genfromtxt("project3_dataset2.txt", usecols = -1, dtype = None)

rows = data_from_file.shape[0]

columns = len(data_from_file[0])


k = 10

splits = 10
#*******************************************************************************

kf = KFold(n_splits=10)

kf.get_n_splits(data_from_file)

total_accuracy = 0

total_precision = 0

total_recall = 0

total_fmeasure = 0

for train_index, test_index in kf.split(data_from_file):

    training_set, testing_set = data_from_file[train_index], data_from_file[test_index]

    labels_training_set, labels_testing_set = labels[train_index], labels[test_index]

    predicted_labels_ret = find_kNN(training_set, labels_training_set, testing_set, labels_testing_set)

    accuracy = calculate_accuracy(predicted_labels_ret, labels_testing_set)

    total_accuracy = total_accuracy + accuracy

    precision = calculate_precision(predicted_labels_ret, labels_testing_set)

    total_precision = total_precision + precision

    recall = calculate_recall(predicted_labels_ret, labels_testing_set)

    total_recall = total_recall + recall

    f1_measure = calculate_fmeasure(predicted_labels_ret, labels_testing_set)

    total_fmeasure = total_fmeasure + f1_measure

    # print accuracy , precision, recall, f1_measure
    #
    # print predicted_labels_ret

print "Accuracy is " , (float(total_accuracy )/ float(splits))*100

print "Precision is " , float(total_precision )/ float(splits)

print "Recall is " , float(total_recall )/ float(splits)

print "F measure is " , float(total_fmeasure)/ float(splits)
