import inspect

def get_function_name():
    """
    Returns the name of the function that calls this function. That is,
    return the function name within itself.
    """
    return inspect.currentframe().f_back.f_code.co_name