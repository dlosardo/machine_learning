"""
Parameter Class
"""


class Parameter(object):
    """
    Constructor
    Parameter has a name and a value.
    Name must be a string.
    Value must be None or a float.
    """
    def __init__(self, name=None, value=None):
        self._name = name
        self._value = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name_value):
        if not isinstance(name_value, basestring):
            raise TypeError("Parameter name must be a string")
        self._name = name_value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value_value):
        #if not isinstance(value_value, float) and value_value is not None:
        #    raise TypeError("Parameter value must be None or a float")
        self._value = value_value

    def is_initialized(self):
        return self.value is not None
