# HyRS
This is for the paper

Tong Wang, "Gaining Free or Low-Cost Transparency with Interpretable Partial Substitute", ICML, 2019

Example:

The input data Xrain needs to be binarized
The target variable Ytrain and Ybtrain needs to be a binary array of 0 or 1

    model = hyb(Xtrain,Ytrain,Ybtrain)
    model.generate_rulespace(supp,4,Nrules, need_negcode = True, method = 'randomforest',criteria = 'precision')
    maps,error,exp = model.train(500, False)
