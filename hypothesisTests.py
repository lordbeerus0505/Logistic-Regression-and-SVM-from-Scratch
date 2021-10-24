
""" 
    Formulate a hypothesis about the performance difference between at least two of the models
    (Any pair of the 3 models can be used to form your hypothesis).
    (v) Test your hypothesis and discuss whether the observed data support your hypothesis (i.e., are
    the observed differences signicant).
"""

# Here we will go ahead and compare SVM with NBC in terms 
# of accuracy (performance difference) while retaining the hyper parameters from before

# Let us come up with a null hypothesis and an alternate hypothesis
# Null Hypothesis - SVM accuracy = LR accuracy
# Alternative Hypothesis - SVM accuracy = LR accuracy

# Note, this is being done for each t_frac value provided.

from scipy.stats import ttest_rel

accuracyNBC = [[0.7024999999999999, 0.011114153511807532], 
[0.7126923076923076, 0.00938256598051398], [0.7182692307692308, 0.009000534172181973],
[0.725, 0.009060937684145957], [0.7232692307692308, 0.007340261430618292],
[0.7244230769230768, 0.007473077912823088]]

accuracySVM = [[0.5834615384615385, 0.01776157011024264],
[0.5340384615384615, 0.013376240748732332], [0.5653846153846154, 0.011760680654938963],
[0.5675, 0.011398023549212485], [0.551923076923077, 0.01071414482860317],
[0.5601923076923077, 0.010103202667570909]]

accuracyLR = [[0.6276923076923075, 0.026723435629318323],
[0.6244230769230767, 0.019047297871693444], [0.6407692307692308, 0.012825063294998202],
[0.6444230769230769, 0.010663804010592618], [0.6430769230769231, 0.009464977486321241],
[0.6578846153846154, 0.009160186879569598]]

accuracyForNBC = [x[0] for x in accuracyNBC]
accuracyForLR = [x[0] for x in accuracyLR]
accuracyForSVM = [x[0] for x in accuracySVM]
# import pdb; pdb.set_trace()
pvalue = ttest_rel(accuracyForLR, accuracyForSVM).pvalue
# The significance is taken as 0.01 or 99%
if pvalue < 0.01:
    print("Rejecting Null Hypothesis; Accepting Alternative Hypothesis")
else:
    print("Accepting Null Hypothesis")