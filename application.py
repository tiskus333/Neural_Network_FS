#!/usr/bin/python3 -i

import os
import sys
import pandas as pd

from NeuralNetwork import *


class DataClass:
    numeric_values = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
                      'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']
    binary_values = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup',
                     'famsup', 'paid', 'higher', 'activities', 'nursery', 'internet', 'romantic']
    nominal_values = ['reason', 'guardian', 'Mjob', 'Fjob']

    def __init__(self):
        math_class_data = pd.read_csv(os.path.abspath(
            '') + "/data/student-mat.csv", delimiter=";")
        port_class_data = pd.read_csv(os.path.abspath(
            '') + "/data/student-por.csv", delimiter=";")
        data = math_class_data.append(port_class_data)
        self.data_raw = data[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
                              'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
                              'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
                              'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
                              'Walc', 'health', 'absences', 'G1', 'G2', 'G3']]
        self.data_numeric_raw = data[self.numeric_values]
        self.data_binary_raw = data[self.binary_values]
        self.data_nominal_raw = data[self.nominal_values]

        self.data_numeric_scaled = self.normal_scaling(self.data_numeric_raw)
        self.data_binary_scaled = self.binary_conversion(self.data_binary_raw)
        self.data_nominal_scaled = self.nominal_conversion(
            self.data_nominal_raw)

    def normal_scaling(self, raw):
        '''
        after this scaling all of columns have distribution N(0,1)
        '''
        scaled = raw.copy()
        for column in raw.columns:
            raw_c = raw[column]
            scaled[column] = (raw_c - raw_c.mean())/raw_c.std()
        return scaled

    def binary_conversion(self, raw):
        scaled = raw.copy()
        for column in scaled.columns:
            values = set(scaled[column].values)
            if len(values) != 2:
                raise AttributeError
            values = list(values)
            d = {values[0]: 0, values[1]: 1}
            scaled[column] = scaled[column].map(d)
        return scaled

    def nominal_conversion(self, raw):
        return pd.get_dummies(raw)
