class Dictionary:
    """
    A class that allows dictionary-like access to its items using dot notation.

    This class wraps a dictionary and provides an interface for accessing,
    setting, and deleting its items using dot notation, similar to attribute access.

    Parameters
    ----------
    dictionary : dict
        A dictionary to be accessed using dot notation.

    Attributes
    ----------
    __dict__ : dict
        The internal dictionary used for storage.

    Methods
    -------
    __getattr__(key)
        Returns the value associated with the attribute key.
    __setattr__(key, value)
        Sets the value for the attribute key.
    __delattr__(key)
        Deletes the attribute key from the dictionary.
    __repr__()
        Returns a string representation of the Dictionary object.

    Examples
    --------
    >>> dic = Dictionary({'A': 76, 'B': 64})
    >>> dic.A
    76
    >>> dic.B
    64

    Modify values
    
    >>> dic.A = 100
    >>> dic.A
    100

    Delete attributes
    
    >>> del dic.B
    >>> dic.__dict__
    {'A': 100}
    """

    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        raise AttributeError(f"'Dictionary' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __delattr__(self, key):
        if key in self.__dict__:
            del self.__dict__[key]
        else:
            raise AttributeError(f"'Dictionary' object has no attribute '{key}'")
    
    def items(self):
        return self.__dict__.items()
    
    def keys(self):
        return self.__dict__.keys()
    
    def values(self):
        return self.__dict__.values()
    
    def __repr__(self):
        return f"Dictionary({self.__dict__.keys()})"
    
    
def dict_none_remove(dct: dict, return_keys: bool = False) -> dict:
    """
    Remove keys with `None` values from a dictionary.

    Parameters
    ----------
    dct : dict
        The dictionary from which to remove keys with `None` values.
    return_keys : bool, optional (default=None)
        If `True`, return a tuple containing the dictionary without 
        `None` values and the first key that had a `None` value.

    Returns
    -------
    result : dict
        - A dictionary with all keys that had `None` values removed.
        - tuple of dict and key, optional
            If `return_keys` is True, return a tuple where the first 
            element is the dictionary with `None` values removed, and the 
            second element is the first key that had a `None` value.
            If no keys had `None` values, return `None` as the second 
            element.

    Examples
    --------
    >>> import stemlab as stm
    
    >>> d = {'a': 1, 'b': None, 'c': 3}
    >>> stm.dict_none_remove(d)
    {'a': 1, 'c': 3}
    
    Returning the first key with a `None` value:
    
    >>> stm.dict_none_remove(d, return_keys=True)
    ({'a': 1, 'c': 3}, 'b')
    
    No keys have `None` values:
    
    >>> d2 = {'x': 10, 'y': 20}
    >>> stm.dict_none_remove(d2, return_keys=True)
    ({'x': 10, 'y': 20}, None)
    """
    none_removed = {
        key: value for key, value in dct.items() if value is not None
    }
    if return_keys:
        keys_with_none_values = [
            key for key, value in dct.items() if value is None
        ]
        result = none_removed, keys_with_none_values[0] if keys_with_none_values else None
    else:
        result = none_removed
    
    return result


def dict_none_keys(dct: dict) -> list:
    """
    Return a list of keys that have `None` values in a dictionary.

    Parameters
    ----------
    dct : dict
        The dictionary to check for `None` values.

    Returns
    -------
    list
        A list of keys that have `None` as their value. If no keys 
        have `None` values, returns an empty list.

    Examples
    --------
    >>> import stemlab as stm
    
    >>> d = {'a': 1, 'b': None, 'c': 3, 'd': None}
    >>> stm.dict_none_keys(d)
    ['b', 'd']
    
    No `None` values in the dictionary:
    
    >>> d2 = {'x': 10, 'y': 20}
    >>> stm.dict_none_keys(d2)
    []
    """
    keys_with_none_values = [key for key, value in dct.items() if value is None]

    return keys_with_none_values
