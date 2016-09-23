# DigitRecognizer
from kaggle competitions(using scikit-learn svm model)

# Description
- The datasets are from https://www.kaggle.com/c/digit-recognizer/data
- This program used the initial svm parameters and got 0.936 accuracy on the test set, which is bad... 
- When I set ```C=1000.0, gamma='auto'```, I got 0.96986 accuracy.

# Requirement
- To avoid MemoryError, please note that python x64 is required.
- Scikit-learn is required. See http://scikit-learn.org/stable/install.html

# Usage
- Download the datasets first.("train.csv" and "test.csv")
- If you wanna train your own model, then run ```python train_svm_model.py [your model filename]```. The default filename is "svm.model"
- If you wanna load a trained model, then run ```python load_svm_model.py [your model filename]```. The default filename is "svm.model"
