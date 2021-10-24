# Implementing Logistic Regression and Support Vector Machines from Scratch
## Comparing Methods for Speed Dating Classication
### Submitted by Abhiram Natarajan PUID 0033250677
### Instructions to run code:
**Requirements**:
- pip install numpy
- pip install pandas
- pip install matplotplib
- pip install scipy
- python 3.6.3

#### Please add dating-full.csv to this list before running it.

### Question 1: Preprocessing
As before, the single quotes, conversion to lower and everything else performed in Assignment 2 preprocess.py have been performed. Only the first 6500 entries have been considered. In addition, one hot encoding is performed after sorting, using the pandas framework. Note that the last feature is left out as a reference, which is on when the rest are all 0s.
> Can be run using the command - `python preprocess-assg3.py`  

Outputs the mapped vectors for each of the classes.

### Question 2: Implement Logistic Regression and Linear SVM
There are two major functions here, one that calls LogisticRegresion class and one that calls SupportVectorMachines class. They perform the necessary opertations that are stated in the question using the required Hyper Parameters such as the regularization factor lambda and the learning rate. Tolerance is used as well. Finally, the number of epochs is limited to 500. Multiple stopping conditions are kept in place. All operations are performed with L2 Regularization in place.
> Can be run using the command `python lr_svm.py trainingSet.csv testSet.csv 1` for Logistic Regression and `python lr_svm.py trainingSet.csv testSet.csv 2` for Linear SVM  
The output provided is the Train and Test accuracy. 

### Question 3: Learning Curves and Performance Comparison
This part brings back NBC classifier from Assignment 2 and requires that we preprocess it using the same approach as earlier. Additionally, only the first 6500 rows are considered. A bin size of 5 is used during binning and discretization is used instead of one hot encoding. 20% of the data is used as test set and the rest as training set in this scenario. 
*Note* Naive Bayes will be using trainingSet_NBC.csv while LR and SVM will continue to use trainingSet.csv. 
> Can be run using the command `python cv.py`. *Note this operation takes time, so print statements are added to show that there is progress*  

The output includes a graph `learningCurves.png` that has the three curves in it. It compares the model input size to its accuracy. Using 10 fold cross validation. Further, for creating and testing a Hypothesis, we use the output arrays from cv.py in `hypothesisTests.py`  
This produces an output of `Null Hypothesis Rejected; Alternative Hypothesis accepted.` When a significance of 0.01 or 99% is used in the p-value.

#### Special conditions
Sometimes, there are warnings thrown about division by zero. These are explictly ignored in the code but can be commented out if required.