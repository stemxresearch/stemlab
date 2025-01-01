import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from stemlab.graphics.imagecode import image_to_base64


def gph_barchart_simple(
    values,
    categories: list = ['Principal', 'Amount', 'Interest'],
    colors: list = ['green', 'teal', 'gold'],
    width: int = 500,
    height: int = 325,
    bargap: float = 0.6
):
    """
    Create an interactive bar graph with custom categories, values, 
    and colors.

    Parameters
    ----------
    values : list of float
        The values to be plotted.
    categories : list of str
        The category names for the x-axis.
    colors : list of str
        The colors for the bars.
    width : int, optional (default=550)
        Width of the graph in pixels.
    height : int, optional (default=400)
        Height of the graph in pixels.
    bargap : float, optional (default=0.6)
        Gap between bars.
    
    Returns
    -------
    None
        Displays the bar graph.
    """
    fig = go.Figure(data=[go.Bar(
        x=categories, 
        y=values, 
        marker_color=colors,
        text=[f'{value:,.0f}' for value in values],
        textposition='outside',
        textangle=0,
        textfont=dict(size=13)
    )])

    fig.update_layout(
        yaxis_title='Value',
        xaxis=dict(
            tickvals=categories,
            ticktext=categories,
            tickangle=0,
            tickmode='array',
            zeroline=True,
            showline=True,
            linecolor='gray',
            linewidth=1,
        ),
        yaxis=dict(
            zeroline=True,
            showline=True,
            linecolor='gray',
            linewidth=1,
        ),
        bargap=bargap,
        width=width,
        height=height,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    image_html = image_to_base64(fig=fig)
    
    return image_html


def gph_scatter(x: list, y: list, **kwargs):
    """
    Plot two vectors with diamond markers and text annotations 
    east of each marker.
    
    Parameters
    ----------
    x : array-like
        The x-coordinates of the points.
    y : array-like
        The y-coordinates of the points.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([10, 11, 12, 13, 14])
    >>> basic_scatter(x=x, y=y)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    figure_args = kwargs
    decimal_points = figure_args.get('decimal_points', 2)
    plt.figure(
        figsize=(figure_args.get('width', 6), figure_args.get('height', 4))
    )
    plt.plot(
        x,
        y,
        marker=figure_args.get('marker', 'D'),
        linestyle=figure_args.get('linestyle', '-'),
        color=figure_args.get('color', 'b')
    )
    plt.title(figure_args.get('title', ''))
    plt.xlabel(figure_args.get('xlabel', ''))
    plt.ylabel(figure_args.get('ylabel', ''))
    
    if figure_args.get('add_labels'):
        for i in range(len(x)):
            plt.text(
                x[i] + figure_args.get('gap_label_x', 0), 
                y[i] + figure_args.get('gap_label_y', 0),
                f'({round(x[i], decimal_points)}, {round(y[i], decimal_points)})',
                fontsize=figure_args.get('fontsize', 9),
                ha=figure_args.get('ha', 'left')
            )
        
    plt.show()