"""
    Learn a NBC model using the data in the training dataset, and then apply the learned model
    to the test dataset.
    Evaluate the accuracy of your learned model and print out the model’s accuracy on both the
    training dataset and the test dataset as specified below. 
"""
import pandas as pd
import numpy

""" 
    Write a function named nbc(t_frac) to train your NBC which takes a parameter t frac that
    represents the fraction of the training data to sample from the original training set. Use the
    sample function from pandas with the parameters initialized as random state = 47, frac =
    t_frac to generate random samples of training data of different sizes.
"""

""" 
    1. Use all the attributes and all training examples in trainingSet.csv to train the NBC by
    calling your nbc(t_frac) function with t_frac = 1. After get the learned model, apply it on all
    examples in the training dataset (i.e., trainingSet.csv) and test dataset (i.e., testSet.csv)
    and compute the accuracy respectively. Please put your code for this question in a file called
    5_1.py.
    Expected output lines:
        Training Accuracy: [training-accuracy-rounded-to-2-decimals]
        Testing Accuracy: [testing-accuracy-rounded-to-2-decimals]
"""
output = []
p_attr = {'positive':{}, 'negative':{}}
p_decision_pos = 0
p_decision_neg = 0
number_of_unique ={}

def retrieveProbabilityCorrect(attr, value):
    # import pdb; pdb.set_trace()
    value_count = 0
    if value in p_attr['positive'][attr]:
        value_count = p_attr['positive'][attr][value]
    # Performing laplacian smoothing as shown in class, only when value_count is 0 do we do this
        if value_count > 0:
            return (value_count)/(p_attr['positive']['decision'][1])
    return (value_count+1.0)/(p_attr['positive']['decision'][1]+len(number_of_unique[attr]))

def retrieveProbabilityIncorrect(attr, value):
    value_count = 0
    if value in p_attr['negative'][attr]:
        value_count = p_attr['negative'][attr][value]
    
    if value_count > 0:
            return (value_count)/(p_attr['negative']['decision'][0])
    return (value_count+1.0)/(p_attr['negative']['decision'][0]+len(number_of_unique[attr]))

def calculateOutputNaiveBayes(dataset):  
    correctProb = 1
    wrongProb = 1

    result = []
    for index, row in dataset.iterrows():
        # p_attr has all attributes for both positive and negative
        for col in p_attr['positive']:
            if col != 'decision':
                correctProb *= retrieveProbabilityCorrect(col, row[col])
                wrongProb *= retrieveProbabilityIncorrect(col, row[col])
        val = 0
        # multiplying by the prior probability to make the comparison since
        # propotional to product of all posterior and the prior
        if correctProb * p_decision_pos > wrongProb * p_decision_neg:
            val = 1

        result.append([val, row['decision']])
        correctProb = 1
        wrongProb = 1
    return result


def findBayesianProbability(pxArr, class_prior_probability, predictor_prior_probability):
    # Given the prior and conditional probabilities, we can find posterior probability
    p_c_given_x = 0
    prod = 1
    for px in pxArr:
        prod *= px
    p_c_given_x = float(prod * class_prior_probability)/ predictor_prior_probability
    return p_c_given_x

def createPxArr(p_attr):
    # import pdb; pdb.set_trace()

    probability_of_attr = {'yes':{}, 'no': {}}
    for attr in p_attr['positive']:
        if attr != 'decision':
            print(attr)
            probability_of_attr['yes'][attr] = p_attr['positive'][attr][1]/(p_attr['positive']['decision'][1])
            probability_of_attr['no'][attr] = p_attr['negative'][attr][0]/(p_attr['negative']['decision'][0])


def nbc(train_Set, test_Set):
    import pprint 
    global p_decision_neg, p_decision_pos, number_of_unique
    
    # basically shuffles the dataset even when t_frac = 1 so useful.

    # need to decide which param to take as prior and calculate.
    # P(decision| all other attr) = PI(P(attr_i|decision))*P(decision)/ P(attr)
    
    for attr in train_Set:
        p_attr['positive'][attr] = train_Set[train_Set['decision']==1][attr].value_counts(sort=False).to_dict()
        # number of positives and negatives for each attr are tabulated
        p_attr['negative'][attr] = train_Set[train_Set['decision']==0][attr].value_counts(sort=False).to_dict()
        # now we know, if decision was 1, what the probability of each attribute could be
        number_of_unique[attr] = train_Set[attr].value_counts(sort=False).to_dict()

    # p_attr['positive']['decision'] returns an ordered pair (1, value), we only want value hence index 1
    p_decision_pos = p_attr['positive']['decision'][1]/(p_attr['positive']['decision'][1] + p_attr['negative']['decision'][0])
    p_decision_neg = p_attr['negative']['decision'][0]/(p_attr['positive']['decision'][1] + p_attr['negative']['decision'][0])
    # findPosterior requires class_prior_probability and predictor_prior_probability
    # pprint.pprint(p_attr)
    # createPxArr(p_attr)
    # import pdb; pdb.set_trace()
    trainingResult = calculateOutputNaiveBayes(train_Set)
    # print('Training Accuracy: %.2f'%findAccuracy(trainingResult))

    testResult = calculateOutputNaiveBayes(test_Set)
    # print('Testing Accuracy: %.2f'%findAccuracy(testResult))
    return findAccuracy(testResult)

def nbc_5_2(t_frac, fileNameTrain, fileNameTest):
    global p_decision_neg, p_decision_pos
    trainingSet = pd.read_csv(fileNameTrain)
    testSet = pd.read_csv(fileNameTest)
    trainingSet = trainingSet.sample(frac=t_frac, random_state = 47)
    testSet = testSet.sample(frac=t_frac, random_state = 47)
    
    # basically shuffles the dataset even when t_frac = 1 so useful.

    # need to decide which param to take as prior and calculate.
    # P(decision| all other attr) = PI(P(attr_i|decision))*P(decision)/ P(attr)
    
    for attr in trainingSet:
        p_attr['positive'][attr] = trainingSet[trainingSet['decision']==1][attr].value_counts(sort=False).to_dict()
        # number of positives and negatives for each attr are tabulated
        p_attr['negative'][attr] = trainingSet[trainingSet['decision']==0][attr].value_counts(sort=False).to_dict()
        # now we know, if decision was 1, what the probability of each attribute could be
        number_of_unique[attr] = trainingSet[attr].value_counts(sort=False).to_dict()


    # p_attr['positive']['decision'] returns an ordered pair (1, value), we only want value hence index 1
    p_decision_pos = p_attr['positive']['decision'][1]/(p_attr['positive']['decision'][1] + p_attr['negative']['decision'][0])
    p_decision_neg = p_attr['negative']['decision'][0]/(p_attr['positive']['decision'][1] + p_attr['negative']['decision'][0])
    # findPosterior requires class_prior_probability and predictor_prior_probability
    trainingResult = calculateOutputNaiveBayes(trainingSet)
    # print(findAccuracy(trainingResult))

    testResult = calculateOutputNaiveBayes(testSet)
    # print(findAccuracy(testResult))
    return findAccuracy(trainingResult), findAccuracy(testResult)


def findAccuracy(list1):
    total = len(list1)
    count = 0
    for i in range(total):
        if list1[i][0] == list1[i][1]:
            count += 1
    return float(count)/total
def main():
    t_frac = 1
    # dataframes are loaded directly inside now to sample by t_frac using pandas
    nbc(t_frac)

if __name__ == '__main__':
    main()

# using this as just calling main() slows down 5_2




