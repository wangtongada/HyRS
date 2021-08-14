# HyRS
This github code is the implementation for the paper

Tong Wang, "Gaining Free or Low-Cost Transparency with Interpretable Partial Substitute", ICML, 2019

Wang, Tong, and Qihang Lin. "Hybrid Predictive Models: When an Interpretable Model Collaborates with a Black-box Model." Journal of Machine Learning Research 22.137 (2021): 1-38.

## Requirements

You'll need to install fim package https://borgelt.net/pyfim.html, which is to generate candidate rules. However, if you cannot install it for some reason, you can use the alternative approach of using random forest to generate rules. In that case, just comment out the "from fim import fpgrowth,fim" line in hybrid.py and choose "random forest" when calling function generate_rulespace(), as shown in example.ipynb.

## How to use the code

The input data Xtrain needs to be binarized into a matrix of 0 and 1, i.e., the categorical features need to be one-hot-encoded and the numeric features need to be discretized. The target variable Ytrain and Ybtrain needs to be a binary array of 0 or 1.

supp is the minimal support, which can be set to 5% or 10%. Nrules is the size of the rule space, which can be set to 5000 or 10000

See the example.ipynb file for an example of how to use the model.

## Contact

I don't actively check github. If you have questions, please email me at tong-wang@uiowa.edu
