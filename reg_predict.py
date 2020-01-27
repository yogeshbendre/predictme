import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pickle

def predict_reg(input_data):
    reg_mdl = pickle.load(open('reg_model.pkl','rb'))
	diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
	diabetes_X = diabetes_X[:, np.newaxis, 2]
    #pr = reg_mdl.predict(input_data)
	pr = reg_mdl.predict(diabetes_X)
    return(str(pr))
