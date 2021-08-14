# HyRS
This is for the paper

Tong Wang, "Gaining Free or Low-Cost Transparency with Interpretable Partial Substitute", ICML, 2019

Example:

The input data Xtrain needs to be binarized into a matrix of 0 and 1, i.e., the categorical features need to be one-hot-encoded and the numeric features need to be discretized. The target variable Ytrain and Ybtrain needs to be a binary array of 0 or 1.

supp is the minimal support, which can be set to 5% or 10%. Nrules is the size of the rule space, which can be set to 5000 or 10000

    model = hyb(Xtrain,Ytrain,Ybtrain)
    model.generate_rulespace(supp,4,Nrules, need_negcode = True, method = 'randomforest',criteria = 'precision')
    maps,error,exp = model.train(500)
