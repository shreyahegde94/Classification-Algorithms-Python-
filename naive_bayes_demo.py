import numpy as np
from collections import Counter, defaultdict
from sklearn.model_selection import KFold


#Gives the prior probabilities of label
def label_prior_probabilites(data):
    samples = len(data)
    label_list = []
    for i in range(samples):
        label_list.append(data[i][-1])
    counter = dict()
    for i in label_list:
        if not i in counter:
            counter[i] = 1
        else:
            counter[i] += 1
    for key in counter.keys():
        counter[key] = float(counter[key])/ float(samples)
    return counter

#This function seperates the data based on the label
def seperate(data):
    seperated_data = defaultdict(list)
    for sample in data:
        seperated_data[sample[-1]].append(sample)
    new_list = seperated_data.values()
    no_of_labels = len(new_list)
    data_dict = dict()
    for i in range(no_of_labels):
        data_dict[i] = new_list[i]
    return data_dict

#Performs naive bayes and classifies it to a label
def naive_bayes(data,sample):
    rows = len(data)
    cols = len(data[0])
    labels = []
    for i in range(rows):
        labels.append(data[i][-1])
    unique_labels = np.unique(labels)
    label_probabiltites = label_prior_probabilites(data)
    print " "
    print "class probabilties", label_probabiltites
    data_by_class = seperate(data)
    length = len(unique_labels)
    i = 0
    result = dict()
    temp_2 = [0] * (cols -1)
    prob_dr = 1.0
    for entry in data:
        for index in range(len(sample)):
            if entry[index] == sample[index]:
                temp_2[index] = temp_2[index] + 1
    for j in range(len(sample)):
        prob_dr = prob_dr * (float(temp_2[j])/float(rows))
    for label in data_by_class:
        prob = 1.0
        temp = [0] * (cols - 1)
        size = len (data_by_class[label])
        for value in data_by_class[label]:
            for t in range(len(sample)):
                if sample[t] == value[t]:
                    temp[t] = temp[t] +1

        for j in range(len(sample)):
            prob = prob * (float(temp[j])/float(size))
        prob = float(label_probabiltites[label] * prob)/float(prob_dr)
        result[i] = prob
        i = i +1
    for key in result:
        print " "
        print "class " , key, "has probabilty" , result[key]
    maximum = 0.0
    answer = 999.0
    for key in result:
        if result[key] > maximum:
            maximum = result[key]
            answer = key
    print " "
    print " The data will get classified to class ", answer
    print " "

#step 1 - format input data

print " "
result_Data = []
with open('project3_dataset4.txt','r') as f:
    for line in f:
        result_Data.append(line.strip().split('\t'))
rows = len(result_Data)
cols = len(result_Data[0])

for i in range(rows):
    for j in range(cols):
        if not(result_Data[i][j].isalpha()):
            result_Data[i][j] = float(result_Data[i][j])

for i in range(rows):
    result_Data[i][-1] = int(result_Data[i][-1])

result_Data = np.array(result_Data, dtype = object)

#step 2 - get input from console

response = raw_input("enter query:")
response_list = []
response_list.append(response.split("/"))
sample = response_list[0]

#step 3 - call naive bayes function to classify given data

naive_bayes(result_Data,sample)
