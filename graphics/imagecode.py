import base64
from IPython.display import HTML

def image_to_base64(fig):
    """
    Convert an image file to a base64 encoded string.

    Parameters
    ----------
    fig : Image
        Path to the image file.

    Returns
    -------
    image_str : str
        The base64 encoded string of the image.
    """
    image_bytes = fig.to_image(format="png")
    encoded_string = base64.b64encode(image_bytes).decode('utf-8')
    image_html = HTML(f'<img src="data:image/png;base64,{encoded_string}" />')
        
    return image_html
