# HyRS
This is for the paper

Tong Wang, "Gaining Free or Low-Cost Transparency with Interpretable Partial Substitute", ICML, 2019

Wang, Tong, and Qihang Lin. "Hybrid Predictive Models: When an Interpretable Model Collaborates with a Black-box Model." Journal of Machine Learning Research 22.137 (2021): 1-38.

bib:
@inproceedings{wang2019gaining,
  title={Gaining free or low-cost interpretability with interpretable partial substitute},
  author={Wang, Tong},
  booktitle={International Conference on Machine Learning},
  pages={6505--6514},
  year={2019},
  organization={PMLR}
}
@article{wang2021hybrid,
  title={Hybrid Predictive Models: When an Interpretable Model Collaborates with a Black-box Model},
  author={Wang, Tong and Lin, Qihang},
  journal={Journal of Machine Learning Research},
  volume={22},
  number={137},
  pages={1--38},
  year={2021}
}
Example:

The input data Xtrain needs to be binarized into a matrix of 0 and 1, i.e., the categorical features need to be one-hot-encoded and the numeric features need to be discretized. The target variable Ytrain and Ybtrain needs to be a binary array of 0 or 1.

supp is the minimal support, which can be set to 5% or 10%. Nrules is the size of the rule space, which can be set to 5000 or 10000

See the example.ipynb file for an example of how to use the model.
