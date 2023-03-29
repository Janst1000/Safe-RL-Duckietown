from sklearn.linear_model import LinearRegression
from pprint import pprint
import numpy as np

import pandas as pd

class robot_model():
    def __init__(self) -> None:
        self.model = LinearRegression()
        self.input_data = []
        self.output_data = []

    def split_arrays_from_file(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        first_arrays = []
        second_arrays = []

        for line in lines:
            array_strings = line.strip().split('/')
            first_arrays.append(np.array(eval(array_strings[0])))
            second_arrays.append(np.array(eval(array_strings[1])))
        
        self.input_data = first_arrays
        self.output_data = second_arrays
    
    def train_model(self):
        self.model.fit(self.input_data, self.output_data)
        print("Coefficients: {}".format(self.model.coef_))
        print("Intercept: {}".format(self.model.intercept_))

    def save_model(self):
        try:
            # save model coefficients and intercepts
            np.savez("./packages/rl_agent/config/model.npz", coef=self.model.coef_, intercept=self.model.intercept_)
        except:
            print("Couldn't save model to /packages/rl_agent/config/model.npz. Saving to current directory instead.")
            np.savez("model.npz", coef=self.model.coef_, intercept=self.model.intercept_)


    def load_model(self):
        # load model coefficients
        self.model.coef_ = np.load("model.npz")["coef"]
        # load model intercept
        self.model.intercept_ = np.load("model.npz")["intercept"]



if __name__ == "__main__":
    robot = robot_model()
    robot.split_arrays_from_file("data.txt")
    robot.train_model()
    robot.save_model()
