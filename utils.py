import pathlib


def get_public_names(globals_dict, exclude_imports=['os', 'pathlib']):
    """
    Get public names from a module, excluding file names, 
    directory names, and specified imports.

    Parameters
    ----------
    globals_dict : dict
        The globals() dictionary of the module.

    exclude_imports : set or list, optional (default=['os', 'pathlib'])
        A set or list of import names to exclude from the output.

    Returns
    -------
    public_names : list
        A list of public names to be exposed by dir().
        
    """
    if exclude_imports is None:
        exclude_imports = set()

    # path to the module's directory
    module_dir = pathlib.Path(globals_dict['__file__']).parent

    # all Python files in the directory
    files = {file.stem for file in module_dir.glob('*.py')}
    
    # all subdirectories in the directory
    dirs = {folder.name for folder in module_dir.iterdir() if folder.is_dir()}

    # all attributes of the module
    all_names = globals_dict.keys()

    # combine all excluded names
    excluded_names = files | dirs | set(exclude_imports)

    # filter out unwanted names
    public_names = [
        name for name in all_names if name not in excluded_names
    ]

    return public_names
