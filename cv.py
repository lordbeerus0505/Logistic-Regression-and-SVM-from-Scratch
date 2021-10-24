""" 
In this part, you are asked to use incremental 10-fold cross validation to plot learning curves for
diferent classiers (NBC, LR, SVM), with training sets of varying size but constant test set size.
You are then asked to compare the performance of dierent classiers given the learning curves.
"""
import numpy
import sys
import pandas as pd
import copy
import importlib
import lr_svm
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fivePart1 = importlib.import_module('5_1')
encoderDict = {}
LRSVMsplits = []
NBCsplits = []
dataLRSVM = 0
dataNBC = 0

def encoder(pd_series):
    # More information here https://pandas.pydata.org/docs/reference/api/pandas.Series.cat.categories.html easier that pandas.factorize
    encoding = {}
    name = pd_series.name
    count = 0
    for category in pd_series.cat.categories:
        encoding[category] = count
        count += 1
    
    encoderDict[name] = encoding
    return pd_series.cat.codes

def read_dataset(fromFile, toFile):
    """ The format of values in some columns of the dataset is not unified. Strip the surrounding
        quotes in the values for columns race, race_o and field (e.g., ‘Asian/Pacific Islander/AsianAmerican’ → Asian/Pacific Islander/Asian-American), count how many cells are changed
        after this pre-processing step, and output this number.
     Expected output line: Quotes removed from [count-of-changed-cells] cells. 0 indexed 
     race - col3
     race_o - col4
     field - col8
    """
    number_of_quotes = 0
    data = pd.read_csv(fromFile)
    data = data.head(6500)
    race = data['race']
    race_o = data['race_o']
    field = data['field']

    # using regex to remove quotes

    number_of_quotes += race.str.count("\'.*\'").sum()
    data['race'] = race.str.strip('\'')

    number_of_quotes += race_o.str.count("\'.*\'").sum()
    data['race_o'] = race_o.str.strip('\'')

    number_of_quotes += field.str.count("\'.*\'").sum()
    data['field'] = field.str.strip('\'')
    
    print ('Quotes removed from %s cells' %number_of_quotes)

    number_of_lowercase_conversions = 0
    
    # Needs to be purely lowercase, so subtracting from total if even 1 char is not lower case.
    # This is done by subtracting all PURE lowercase words
    number_of_lowercase_conversions += len(data) - data['field'].str.islower().sum()
    data['field'] = data['field'].str.lower()
    print("Standardized %s cells to lower case"%(number_of_lowercase_conversions))

    data['gender'] = encoder(data['gender'].astype('category'))
    data['race'] = encoder(data['race'].astype('category'))
    data['race_o'] = encoder(data['race_o'].astype('category'))
    data['field'] = encoder(data['field'].astype('category'))

    print("Value assigned for male in column gender: %s"% encoderDict['gender']['male'])
    print("Value assigned for European/Caucasian-American in column race: %s"% encoderDict['race']['European/Caucasian-American'])
    print("Value assigned for Latino/Hispanic American in column race_o: %s"% encoderDict['race_o']['Latino/Hispanic American'])
    print("Value assigned for law in column field: %s"% encoderDict['field']['law'])

    importance = ['attractive_important', 'sincere_important', 'intelligence_important', 'funny_important', 'ambition_important', 'shared_interests_important']
    partner_metrics = ['pref_o_attractive','pref_o_sincere','pref_o_intelligence','pref_o_funny','pref_o_ambitious','pref_o_shared_interests']

    sum_importance = 0
    for i in importance:
        sum_importance += data[i]
    sum_partner_metrics = 0
    for i in partner_metrics:
        sum_partner_metrics += data[i]
    
    # most sum up to 100 but not all, divding all by the sum so % will be the same.

    for i in range(len(importance)):
        # both importance and partner_metrics are of length 6
        data[importance[i]] = data[importance[i]]/sum_importance
        data[partner_metrics[i]] = data[partner_metrics[i]]/sum_partner_metrics

    sum_importance_arr = []
    for i in importance:
        sum_importance_arr.append(data[i].sum())
    sum_partner_metrics_arr = []
    for i in partner_metrics:
        sum_partner_metrics_arr.append(data[i].sum())

    size = len(data)
    # Finding the mean value of these params. Take sum of the entire column at once and then divide by length
    for i in range(6):
        print ('Mean of %s : %s' %(importance[i],round(sum_importance_arr[i]/size, 2)))
    for i in range(6): 
        print ('Mean of %s : %s' %(partner_metrics[i],round(sum_partner_metrics_arr[i]/size, 2)))

    # Writing preprocessed data to dating.csv To prevent first column of serial numbers, using index = False
    data.to_csv(toFile, index = False, mode = 'w')

def part_a(data_frame, outputFile):
    otherCols = ['sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 
    'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies', 'concerts', 'music', 
    'shopping', 'yoga' , 'expected_happy_with_sd_people', 'like', 'interests_correlate'
    ]
    # Rest of the fields have already been handled for and seem fine. All of these have 
    # a range 0-10 (except interest corr) so check max and min

    improper_data_list = []

    for col in otherCols[:-1]:
        maxi = data_frame[col].max()
        mini = data_frame[col].min()
        improper_data_list.append([mini, maxi])
    
    max_interests = data_frame[otherCols[len(otherCols)-1]].max()
    min_interests = data_frame[otherCols[len(otherCols)-1]].min()

    issue_data = []

    if (max_interests > 1.0 or min_interests < -1.0):
        issue_data.append(len(otherCols)-1)
    
    for i in range(len(improper_data_list)):
        if improper_data_list[i][0]<0 or improper_data_list[i][1]>10:
            issue_data.append(i)

    #issue_data has 7,9 so gaming and reading have max issues. The values correspond to 14 and 13 respectively.
    for i in [7,9]:
        col = otherCols[i]
        # for every row where data_frame[col]>10
        data_frame.loc[data_frame[col]>10, col] = 10

    discrete_cols = ['gender', 'race', 'race_o', 'samerace', 'field', 'decision'] 
    attributes = ['pref_o_attractive','pref_o_sincere','pref_o_intelligence','pref_o_funny','pref_o_ambitious',
    'pref_o_shared_interests', 'attractive_important', 'sincere_important', 'intelligence_important', 'funny_important',
    'ambition_important', 'shared_interests_important'] 

    for col in data_frame:
        if col not in discrete_cols:
            # Hasn't already been binned

            if col in attributes:
                bin_range = numpy.arange(0,1.001,0.2)
            elif col in ['age', 'age_o']:
                bin_range = numpy.arange(18,58.001,8)
            elif col == 'interests_correlate':
                # if using -1.0001 instead, values slightly change [18, 713, 2498, 2883, 632]
                bin_range = numpy.arange(-1.00,1.001,0.4)
            else:
                bin_range = numpy.arange(0,10.01,2) 
                # 5 bins for all other kinds which are 
                #in the 0-10 range - basically otherCols

            # Using pd.cut as cut splits as equal width bins opposed to qcut which is equal frequency bins
            data_frame[col] = pd.cut(data_frame[col], bin_range, labels = numpy.arange(5), 
            include_lowest = True, retbins = False)
            
            # Final list is sorted by default but pdf shows like:[] is not sorted in 
            # terms of frequencies, hence not sorting
            print("%s:"%col, data_frame[col].value_counts(sort=False).to_list())
    data_frame.to_csv(outputFile, index = False, mode = 'w')
    return data_frame


def preprocessNBC():
    read_dataset('dating-full.csv', 'dating.csv')

def discretizeNBC():
    # Using a bin size of 5 as instructed
    data = pd.read_csv('dating.csv')
    data = part_a(copy.deepcopy(data), 'datingNBCdiscretize.csv')

def creatingNBCdataset():
    data = pd.read_csv('datingNBCdiscretize.csv')
    test_data = data.sample(frac=0.2, random_state=25)
    indexUsed = test_data.index
    data.drop(indexUsed, axis=0, inplace=True)
    test_data.to_csv("testSet_NBC.csv", index = False)
    data.to_csv("trainingSet_NBC.csv", index = False)

def sampling():
    # The training sets are now trainingSet.csv and trainingSet_NBC.csv
    """ 
        Use the sample function from pandas with the parameters initialized as random state =
        18, frac = 1 to shuffle the training data. Then partition the training data into 10 disjoint
        sets S = [S1... S10], where S1 contains training samples with index from 1 to 520 (i.e., the
        rest 520 lines of training samples after shuffling), and S2 contains samples with index from
        521 to 1040 (i.e., the second 520 lines of training samples after shuffling) and so on. Each set
        has 520 examples.
    """
    global dataLRSVM, dataNBC
    dataLRSVM = pd.read_csv('trainingSet.csv')
    dataNBC = pd.read_csv('trainingSet_NBC.csv')
    # Using frac=1 performs only shuffling of the data in dataset
    dataLRSVM.sample(frac=1, random_state=18)
    dataNBC.sample(frac=1, random_state=18)

    # Now partitioning the data into 10 sets

    size = len(dataNBC)//10
    for i in range(10):
        NBCsplits.append(dataNBC.iloc[size*i:size*(i+1)])

    size = len(dataLRSVM)//10
    for i in range(10):
        LRSVMsplits.append(dataLRSVM.iloc[size*i:size*(i+1)])

def plot(accuracyNBC, accuracyLR, accuracySVM, fracArr, size):
    # import pdb; pdb.set_trace()
    accuracyForNBC = [ x[0] for x in accuracyNBC]
    accuracyForLR = [ x[0] for x in accuracyLR]
    accuracyForSVM = [ x[0] for x in accuracySVM]

    standardErrorNBC = [ x[1] for x in accuracyNBC]
    standardErrorLR = [ x[1] for x in accuracyLR]
    standardErrorSVM = [ x[1] for x in accuracySVM]

    fig = plt.figure()
    fig.set_figwidth(7)
    fig.set_figheight(5)
    fig.subplots_adjust(bottom=0.3)
    plt.errorbar(numpy.multiply(fracArr,size), accuracyForNBC, yerr=standardErrorNBC, color='red')
    plt.errorbar(numpy.multiply(fracArr,size), accuracyForLR, yerr=standardErrorLR, color='blue')
    plt.errorbar(numpy.multiply(fracArr,size), accuracyForSVM, yerr=standardErrorSVM, color='green')

    plt.scatter(numpy.multiply(fracArr,size), accuracyForNBC, color='red')
    plt.scatter(numpy.multiply(fracArr,size), accuracyForLR, color='blue')
    plt.scatter(numpy.multiply(fracArr,size), accuracyForSVM, color='green')

    plt.xlabel('Dataset size')
    plt.ylabel('Accuracy of test')
    red_patch = mpatches.Patch(color='red', label='NBC')
    blue_patch = mpatches.Patch(color='blue', label='LR')
    green_patch = mpatches.Patch(color='green', label='SVM')
    plt.legend(handles=[red_patch, blue_patch, green_patch])
    plt.savefig('learningCurves.png')
    plt.show()

def kfold():
    global dataLRSVM, dataNBC
    accuracyNBC = []
    accuracyLR = []
    accuracySVM = []
    for t_frac in [0.025, 0.05, 0.075, 0.1, 0.15, 0.2]:
        # First for NBC
        accuracy = []
        for idx in range(10):
            test_Set = NBCsplits[idx]
            train_Set = dataNBC.drop(NBCsplits[idx].index) 
            train_Set = train_Set.sample(frac=t_frac, random_state=32)
            print('NBC with index %s, and t_frac %s has a test accuracy of %s'%(idx, t_frac, fivePart1.nbc(train_Set, test_Set)))
            # Calculating average accuracy
            accuracy.append(fivePart1.nbc(train_Set, test_Set))
        accuracyNBC.append([numpy.mean(accuracy), numpy.std(accuracy)/sqrt(10)])

    for t_frac in [0.025, 0.05, 0.075, 0.1, 0.15, 0.2]:
        # Now for LR
        accuracy = []
        for idx in range(10):
            test_Set = LRSVMsplits[idx]
            train_Set = dataLRSVM.drop(LRSVMsplits[idx].index) 
            train_Set = train_Set.sample(frac=t_frac, random_state=32)
            print('LR with index %s, and t_frac %s has a test accuracy of %s'%(idx, t_frac, lr_svm.lr_crossValidate(train_Set, test_Set)))
            accuracy.append(lr_svm.lr_crossValidate(train_Set, test_Set))
        accuracyLR.append([numpy.mean(accuracy), numpy.std(accuracy)/sqrt(10)]) 
          
    for t_frac in [0.025, 0.05, 0.075, 0.1, 0.15, 0.2]:
        # Finally for SVM
        accuracy = []
        for idx in range(10):
            test_Set = LRSVMsplits[idx]
            train_Set = dataLRSVM.drop(LRSVMsplits[idx].index) 
            train_Set = train_Set.sample(frac=t_frac, random_state=32)

            print('SVM with index %s, and t_frac %s has a test accuracy of %s'%(idx, t_frac, lr_svm.svm_crossValidate(train_Set, test_Set)))
            accuracy.append(lr_svm.svm_crossValidate(train_Set, test_Set))
        accuracySVM.append([numpy.mean(accuracy), numpy.std(accuracy)/sqrt(10)]) 

    # import pdb; pdb.set_trace()
    plot(accuracyNBC, accuracyLR, accuracySVM, [0.025, 0.05, 0.075, 0.1, 0.15, 0.2], 4680)

if __name__ == '__main__':
    preprocessNBC()
    discretizeNBC()
    creatingNBCdataset()
    # Begin 10 fold cross validation
    sampling()
    # K Fold Cross Validation step
    kfold()

