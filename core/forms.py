def selected_option(selected: str) -> str:
    """
    Extract the value within parentheses from the given string.

    Parameters
    ----------
    selected : str
        The string containing parentheses.

    Returns
    -------
    str
        The value within parentheses.
        
    """
    try:
        if '(' in selected:
            return selected[selected.find("(") + 1:selected.find(")")]
    except Exception:
        return selected