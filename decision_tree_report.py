import numpy as np
from collections import Counter
from sklearn.model_selection import KFold
import time
from sklearn import tree

start_time = time.time()
def count_occurences(length,child):
	global unique_classes
	score = 0.0
	for labels in unique_classes:
		count = 0.0
		for label_child in child:
			if (label_child[-1] == labels):
				count = count + 1
		temp_value = count / length
		score = score + (temp_value * temp_value)
	return score


def gini_index(left,right):

    total_length = float(len(left)+len(right))

    complete_list = []
    complete_list.append(left)

    complete_list.append(right)

    gini =0.0
    for child in complete_list:
        length = float(len(child))
        if length == 0.0:
			continue
        score = count_occurences(length,child)
        gini += (1.0 - score) * (length / total_length)
    return gini


def test_split_condition(column,value,data):

    left = list()
    right = list()
    if(isinstance (value,float)):

        for sample in data:
            if sample[column] < value:

                left.append(sample)
            else:

                right.append(sample)
    elif(isinstance (value,str)):

        for sample in data:
            if sample[column] == value:
                left.append(sample)
            else:
                right.append(sample)

    return left,right

def get_best_split(data_split):

    global unique_classes
    left_child = None
    right_child = None
    column_selected, value_selected, gini_final = 999, 999, 999
    no_of_features = len(data_split[0]) - 1

    check = {}

    for i in range(no_of_features):

        for sample in data_split:

            left, right = test_split_condition(i,sample[i],data_split)
            gini = gini_index(left,right)
            if gini < gini_final:
                left_child = left
                right_child = right
                column_selected = i
                value_selected = sample[i]
                gini_final = gini
    check = {'column_selected':column_selected, 'value':value_selected, 'left_child':left_child, 'right_child':right_child}

    return {'column_selected':column_selected, 'value':value_selected, 'left_child':left_child, 'right_child':right_child}


def leaf_node(data):
	k_labels = list()
	for sample in data:
		k_labels.append(sample[-1])
	count = Counter(k_labels)
	majority_label = count.most_common()[0][0]
	return majority_label



def split_tree(data):
    left = data['left_child']
    right = data['right_child']

    data.pop("left_child",None)
    data.pop("right_child",None)

    if not left or not right:
        data['left'] = data['right'] = leaf_node(left + right)
        return
    data['left'] = get_best_split(left)
    split_tree(data['left'])
    data['right'] = get_best_split(right)
    split_tree(data['right'])


def build_tree(data):
    root = get_best_split(data)

    split_tree(root)
    return root

# Print a decision tree
def print_tree(data,node,depth):
    indent = "  " * depth
    if isinstance(data, dict):
        if isinstance(data['value'],float):
            print('%s%s%d[X%d < %.3f]' % (indent,node,depth,(data['column_selected']+1), data['value']))
            print_tree(data['left'],"left",depth+1)
            print_tree(data['right'],"right",depth+1)
        elif isinstance(data['value'],str):
            print('%s%s%d[X%d == %s]' % (indent,node,depth,(data['column_selected']+1), data['value']))
            print_tree(data['left'],"left",depth+1)
            print_tree(data['right'],"right",depth+1)

    else:
        print('%s[%s]' % (indent,(data)))



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


def predict_label(node,row):

    if isinstance(node['value'], float):
        if row[node['column_selected']] < node['value']:

            if isinstance(node['left'], dict):

                return predict_label(node['left'], row)
            else:
                return node['left']
        else:

            if isinstance(node['right'], dict):

                return predict_label(node['right'], row)
            else:
                return node['right']
    else:

        if row[node['column_selected']] == node['value']:

            if isinstance(node['left'], dict):

                return predict_label(node['left'], row)
            else:
                return node['left']
        else:

            if isinstance(node['right'], dict):

                return predict_label(node['right'], row)
            else:
                return node['right']


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
dataset = result_Data

labels = result_Data[:,-1]

unique_classes, count_of_each_class = np.unique(result_Data[:,-1], return_counts = True)



#*******************************************************************************
splits = 10

kf = KFold(n_splits=10)

kf.get_n_splits(dataset)

total_accuracy = 0

total_precision = 0

total_recall = 0

total_fmeasure = 0

for train_index, test_index in kf.split(dataset):

    predicted_labels_ret = []

    training_set, testing_set = dataset[train_index], dataset[test_index]

    labels_training_set, labels_testing_set = labels[train_index], labels[test_index]

    testing_set = np.delete(testing_set, -1, axis = 1)

    training_tree = build_tree(training_set)


    for test_data in testing_set:


        predicted_labels_ret.append(predict_label(training_tree,test_data))

    accuracy = calculate_accuracy(predicted_labels_ret, labels_testing_set)

    total_accuracy = total_accuracy + accuracy

    precision = calculate_precision(predicted_labels_ret, labels_testing_set)

    total_precision = total_precision + precision

    recall = calculate_recall(predicted_labels_ret, labels_testing_set)

    total_recall = total_recall + recall

    f1_measure = calculate_fmeasure(predicted_labels_ret, labels_testing_set)

    total_fmeasure = total_fmeasure + f1_measure



print "Accuracy is" , (float(total_accuracy )/ float(splits))*100

print "Precision is ", float(total_precision )/ float(splits)

print "Recall is " , float(total_recall )/ float(splits)

print "Fmeasure is " ,float(total_fmeasure)/ float(splits)
#print "print_tree"
#print_tree(training_tree,"root",1)
print "My program took", time.time() - start_time, "to run"
