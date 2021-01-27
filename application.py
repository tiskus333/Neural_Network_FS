#!/usr/bin/python3 -i

'''
Load convert and normalize data to proper format for our network
Pawel Kotiuk
Mateusz Chrusciel
'''

import os
import sys
import pandas as pd

from NeuralNet import *


class DataClass:
    numeric_values_names = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
                      'famrel', 'freetime', 'goout', 'health', 'absences', 'G1', 'G2', 'G3']
    binary_values_names = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup',
                     'famsup', 'paid', 'higher', 'activities', 'nursery', 'internet', 'romantic']
    nominal_values_names = ['reason', 'guardian', 'Mjob', 'Fjob']
    output_values_names= ['Dalc', 'Walc']

    def __init__(self):
        math_class_data = pd.read_csv(os.path.abspath(
            '') + "/data/student-mat.csv", delimiter=";")
        port_class_data = pd.read_csv(os.path.abspath(
            '') + "/data/student-por.csv", delimiter=";")
        data = math_class_data.append(port_class_data)
        data = data.reset_index()
        self.data_raw = data[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
                              'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
                              'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
                              'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
                              'Walc', 'health', 'absences', 'G1', 'G2', 'G3']]
        self.data_numeric_raw = data[self.numeric_values_names]
        self.data_binary_raw = data[self.binary_values_names]
        self.data_nominal_raw = data[self.nominal_values_names]

        self.data_numeric_scaled = self.normal_scaling(self.data_numeric_raw)
        self.data_binary_scaled = self.binary_conversion(self.data_binary_raw)
        self.data_nominal_scaled = self.nominal_conversion(
            self.data_nominal_raw)
        self.data_to_network = pd.concat(
            [self.data_numeric_scaled, self.data_binary_scaled, self.data_nominal_scaled], axis=1)
        self.output_walc = pd.get_dummies(data["Walc"])
        self.output_dalc = pd.get_dummies(data["Dalc"])

    @staticmethod
    def normal_scaling(raw):
        '''
        after this scaling all of columns have distribution N(0,1)
        '''
        scaled = raw.copy()
        for column in raw.columns:
            raw_c = raw[column]
            scaled[column] = (raw_c - raw_c.mean())/raw_c.std()
        return scaled

    @staticmethod
    def binary_conversion(raw):
        scaled = raw.copy()
        for column in scaled.columns:
            values = set(scaled[column].values)
            if len(values) != 2:
                raise AttributeError
            values = list(values)
            d = {values[0]: 0, values[1]: 1}
            scaled[column] = scaled[column].map(d)
        return scaled

    @staticmethod
    def nominal_conversion(raw):
        return pd.get_dummies(raw)

    def get_num_of_rows(self):
        return self.data_to_network.shape

    def get_num_of_columns(self):
        a, b = self.data_to_network.shape
        return b
