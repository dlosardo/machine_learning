"""
Data Class
"""
from numpy import ndarray

class Data(object):
    """Constructor
    """
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data_value):
        if not isinstance(data_value, ndarray):
            raise TypeError("Must be a numpy array object")
        self._data = data_value
