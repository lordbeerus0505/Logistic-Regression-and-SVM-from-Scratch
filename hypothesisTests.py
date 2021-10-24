
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
# Alternative Hypothesis - SVM accuracy != LR accuracy

# Note, this is being done for each t_frac value provided.

from scipy.stats import ttest_rel

accuracyNBC = [[0.6909615384615384, 0.007929282483936309],
[0.7138461538461538, 0.007628582192792404], [0.7257692307692308, 0.0068812841102454956],
[0.7342307692307692, 0.006092239704360343], [0.7392307692307692, 0.006701585081422923],
[0.7469230769230769, 0.006039808766908805]]

accuracySVM = [[0.5569230769230769, 0.01481942592057414],
[0.5538461538461539, 0.01044498124153867], [0.5573076923076923, 0.009100848520164599],
[0.5613461538461537, 0.006363321766603147], [0.578653846153846, 0.008179070200769295],
[0.571346153846154, 0.004729752861427066]]

accuracyLR = [[0.6534615384615384, 0.019731556439536824],
[0.6498076923076923, 0.010555748034509312], [0.6632692307692307, 0.011145388039732529],
[0.6801923076923078, 0.006617452717879647], [0.6501923076923077, 0.007284629599531255],
[0.6605769230769231, 0.009057876039292832]]

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