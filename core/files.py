import os
import datetime


def file_path_name(file_path: str | None = None) -> str:
    """
    Validate file path and name.

    Parameters
    ----------
    file_path : str, optional (default=None)
        The path and name of the file. If not provided, a default path
        will be generated based on the current working directory and
        timestamp, resulting in a unique file name.

    Returns
    -------
    str
        The validated file path.

    Raises
    ------
    OSError
        If the specified file path cannot be created or does not exist.
    """
    if file_path is None:
        date_time = datetime.now().strftime('%Y-%m-%d %H.%M.%S')
        file_path = f'{os.getcwd()}/results-{date_time}.tex'
    else:
        file_path = str(file_path)
        if '/' in file_path:
            file_path, file_name = file_path.rsplit(sep='/', maxsplit=1)
        else:
            file_name = file_path # what is given is a file name without path
            file_path = os.getcwd()
        if not os.path.exists(file_path):
            try:
                os.makedirs(file_path)
                file_path = f'{file_path}/{file_name}'
            except OSError:
                raise OSError(f"Couldn't create the folder '{file_path}'")
        # check that the file path exists
        if not os.path.exists(file_path):
            raise OSError(
                f"The file path '{file_path}' does not exist on "
                "this computer")
            
    return file_path