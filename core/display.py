from typing import Any

from IPython.display import display
from IPython.core.display import Image, Latex, HTML
from pandas import DataFrame

from stemlab.core.htmlatex import tex_display_latex
from stemlab.core.decimals import fround
from stemlab.core.arraylike import conv_to_arraylike, list_join


class Result:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def __repr__(self):
        # return f"Result({self.__dict__})"
        key_value = "\n".join(f"    {key}: {value}" for key, value in self.__dict__.items())
        return f"Result(\n{key_value}\n)"
        
    def __getattr__(self, name: str) -> Any:
        result = self.__dict__.get(name, None)
        if result is None:
            attributes = list_join(
                lst = list(self.__dict__.keys()), html_tags=False
            )
            raise AttributeError(
                f"'dict' object has no attribute '{name}', valid attributes "
                f"are: {attributes}"
            )
        return result


def display_results(result_dict):
    
    decimal_points = result_dict.get('decimal_points', -1)
    for key, value in result_dict.items():
        try:
            value = fround(x=value, decimal_points=decimal_points)
        except:
            pass
        
        if isinstance(value, (DataFrame, Image, Latex)) or 'IPython' in str(value):
            display(value)
        else:
            try:
                if key != 'decimal_points' and value is not None:
                    tex_display_latex(lhs=[key], rhs=[value])   
            except:
                display(value)


def display_latex(html_list: list | str) -> None:
    """
    Write HTML content to a file.

    Parameters
    ----------
    html_list : {str, list}
        A string or a list of strings representing HTML content to be 
        written to the file.
    
    Returns
    -------
    None
    """
    html_list = conv_to_arraylike(
        array_values=html_list, includes_str=True, par_name='html_list'
    )
    
    latex_list = []
    for html_line in html_list:
            latex_list.append(f'\t\t<p>\n\t\t\t{html_line}\n\t\t</p>\n\n')
    latex_str = ''.join(map(str, latex_list))
    
    display(HTML(data=latex_str))