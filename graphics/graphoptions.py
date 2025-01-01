from io import BytesIO
import base64

from numpy import ndarray, round, asfarray, linspace
from matplotlib.pyplot import (
    axis, xlabel, ylabel, tick_params, xticks, yticks, title, text, axhline, 
    axvline, legend, grid, minorticks_on, savefig, style
)
from sympy import sympify, Matrix, flatten

from stemlab.core.arraylike import arr_abrange
import stemlab as mstm


def titles_break(title_str: str) -> str:
    """
    Format a string containing multiple titles separated by '<<>>'.

    Parameters
    ----------
    title_str : str
        The string containing titles separated by '<<>>'.

    Returns
    -------
    title_str : str
        The formatted string with each title on a separate line.

    Examples
    --------
    >>> titles_break("Title1 <<>> Title2 <<>> Title3")
    'Title1\nTitle2\nTitle3'

    >>> titles_break("Title1<<>>Title2<<>>Title3")
    'Title1\nTitle2\nTitle3'

    >>> titles_break("  Title1  <<>>  Title2  <<>>  Title3  ")
    'Title1\nTitle2\nTitle3'

    >>> titles_break("")
    ''
    """
    # Clean up the title string
    xaxis_title = (
        title_str.replace(" <<>>", "<<>>")\
        .replace("<<>> ", "<<>>")\
        .replace("  ", " ")\
        .lstrip()\
        .rstrip()
    )
    
    # Split the string into separate lines where '<<>>' appears
    if xaxis_title:
        return "\n".join(xaxis_title.split("<<>>"))
    return title_str


def check_color(color_str: str, color_name: str = "black") -> str:
    """
    Check if a color for plotting is valid.

    Parameters
    ----------
    color_str : str
        The color string to be checked.
    color_name : str, optional (default='black')
        The default color name to return if the provided color string is not 
        valid.

    Returns
    -------
    str
        The validated color string.

    Examples
    --------
    >>> ValidateArgs.check_color("red")
    'red'

    >>> ValidateArgs.check_color("blah")
    'black'

    >>> ValidateArgs.check_color("blue", "gray")
    'blue'
    """
    if not mstm.is_color_like(color_str):
        return color_name
    return color_str


class FigureCustomizer:
    def __init__(
        self,
        x,
        y,
        text_labels=None,
        x_axis=None,
        y_axis=None,
        figure_title=None,
        figure_caption=None,
        figure_note=None,
        figure_legend=None,
        figure_others=None,
    ):
        self.x = x
        self.y = y
        self.text_labels = text_labels
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.figure_title = figure_title
        self.figure_caption = figure_caption
        self.figure_note = figure_note
        self.figure_legend = figure_legend
        self.figure_others = figure_others

    def customize(self):
        if self.text_labels is not None:
            self._add_text_labels()

        if self.x_axis is not None:
            self._customize_x_axes(self.x, self.x_axis)

        if self.y_axis is not None:
            self._customize_y_axes(self.y, self.y_axis)

        if self.figure_title is not None:
            self._add_title(self.figure_title)

        if self.figure_caption is not None:
            self._add_caption(self.figure_caption)

        if self.figure_note is not None:
            self._add_note(self.figure_note)

        if self.figure_legend is not None:
            self._add_legend(self.figure_legend)

        if self.figure_others is not None:
            self._customize_others(self.figure_others)

        return self.generate_html()


    def _add_text_labels(self):
        """xxxx"""
        x, y, text_labels = self.x, self.y, self.text_labels

        text_labels_text = text_labels["text_labels_text"]
        text_box_style = text_labels["text_box_style"]
        x_labels = text_labels["x_labels"]
        y_labels = text_labels["y_labels"]
        label_font_name = text_labels["label_font_name"]
        label_font_color =  mstm.ValidateArgs.check_color(
            text_labels["label_font_color"], color_name="black"
        )
        label_font_size = text_labels["label_font_size"]
        label_font_style = text_labels["label_font_style"]
        label_font_weight = text_labels["label_font_weight"]
        label_halign = text_labels["label_halign"]
        label_valign = text_labels["label_valign"]
        label_angle = text_labels["label_angle"]
        label_box_background = text_labels["label_box_background"]
        label_box_outline = text_labels["label_box_outline"]
        label_opacity = text_labels["label_opacity"]
        label_decimals = text_labels["label_decimals"]

        try:
            coords_value = sympify(
                text_labels_text.replace(", ", ",")\
                .replace("; ", ";")\
                .replace(" ", "_")\
                .split(";")
            )
            add_text = True
        except:
            add_text = False

        if add_text:
            for p in coords_value:
                text(
                    p[0],
                    p[1],
                    str(p[2]).replace("_", " "),
                    color= mstm.ValidateArgs.check_color(label_font_color, color_name="#999"),
                    fontsize=label_font_size,
                    fontfamily=label_font_name,
                    horizontalalignment=label_halign,
                    verticalalignment=label_valign,
                    rotation=label_angle,
                    bbox=dict(
                        facecolor= mstm.ValidateArgs.check_color(
                            label_box_background, color_name="#green"
                        ),
                        linewidth=label_box_outline,
                        boxstyle=text_box_style,
                        alpha=label_opacity,
                    ),
                    fontstyle=label_font_style,
                    fontweight=label_font_weight,
                )

        add_labels = True
        for k in range(len(x)):
            if x_labels and not y_labels:
                text = f" {round(x[k], label_decimals)} "
            elif not x_labels and y_labels:
                text = f" {round(y[k], label_decimals)} "
            elif x_labels and y_labels:
                text = f" {round(x[k], label_decimals)}, {round(y[k], label_decimals)} "
            else:  # none is selected
                add_labels = False

            if len(x) > 25:
                add_labels = False

            # add labels
            if add_labels:
                text(
                    x[k],
                    y[k],
                    text,
                    color= mstm.ValidateArgs.check_color(label_font_color, color_name="#999"),
                    fontsize=label_font_size,
                    fontfamily=label_font_name,
                    horizontalalignment=label_halign,
                    verticalalignment=label_valign,
                    rotation=label_angle,
                    bbox=dict(
                        facecolor= mstm.ValidateArgs.check_color(
                            label_box_background, color_name="green"
                        ),
                        linewidth=label_box_outline,
                        boxstyle=text_box_style,
                        alpha=label_opacity,
                    ),
                    fontstyle=label_font_style,
                    fontweight=label_font_weight,
                )

    def _customize_x_axes(self):
        """Customize axes"""
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tick_params.html
        x, x_axis = self.x, self.x_axis

        if x_axis is not None:
            if x_axis["xaxis_show_labels"]:
                xaxis_title = mstm.titles_break(title_str=x_axis["xaxis_title"])
                xlabel(
                    xlabel=xaxis_title,
                    labelpad=x_axis["xaxis_title_topmargin"],
                    fontdict=x_axis["labels_font_properties"],
                    loc=x_axis["xaxis_title_location"],
                    verticalalignment=x_axis["xaxis_title_valign"],
                )

                tick_params(
                    axis=x_axis["xaxis_labels_axis"],  # 'x', 'y', 'both'
                    which="major",  #'major', 'minor', 'both'
                    reset=False,
                    direction=x_axis["xaxis_labels_tick_rules"],  # 'in', 'out', 'inout'
                    length=x_axis["xaxis_labels_ticks_length"],
                    width=x_axis["xaxis_labels_ticks_width"],
                    pad=x_axis["xaxis_labels_margin"],
                    labelsize=x_axis["xaxis_labels_font_size"],
                    colors= mstm.ValidateArgs.check_color(x_axis["xaxis_labels_ticks_color"]),
                    labelcolor= mstm.ValidateArgs.check_color(x_axis["xaxis_labels_font_color"]),
                    top=x_axis["xaxis_labels_top"],
                    labeltop=x_axis["xaxis_labels_top"],
                    labelrotation=x_axis["xaxis_labels_angle"],
                )

                # customize ticks
                if x_axis["xaxis_labels_ticks_customize"].lower() != "default":
                    #'Min, max and interval', 'Number of ticks', 'Custom'
                    a, b = asfarray([min(x), max(x)])
                    decimal_points = x_axis["xaxis_labels_decimals"]
                    ticks_customize_str = x_axis["xaxis_labels_ticks_customize"].lower()
                    if ticks_customize_str == "min and max":
                        ticks_values = round(asfarray([a, b]), decimal_points)
                        ticks_labels = ticks_values
                    elif ticks_customize_str == "min, max and interval":
                        try:
                            h = float(sympify(x_axis["xaxis_labels_ticks_value"]))
                        except:
                            h = (b - a - 1) / 10
                        ticks_values = round(arr_abrange(a, b, h), decimal_points)
                        ticks_labels = ticks_values
                    elif ticks_customize_str == "number of ticks":
                        n = int(sympify(x_axis["xaxis_labels_ticks_value"]))
                        ticks_values = round(
                            linspace(a, b, n), decimal_points
                        )
                        ticks_labels = ticks_values
                    elif ticks_customize_str == "custom":
                        values_labels = Matrix(
                            sympify(x_axis["xaxis_labels_ticks_value"])
                        )
                        ticks_values = flatten(values_labels[:, 0])
                        ticks_labels = flatten(values_labels[:, 1])
                    try:
                        xticks(ticks_values, ticks_labels)
                    except:
                        pass

                # reference lines
                if x_axis["xaxis_axes_add_ref"]:
                    try:
                        ref_points = flatten(
                            Matrix(sympify(x_axis["xaxis_axes_add_ref"]))
                        )
                        for point in ref_points:
                            axvline(
                                x=point,
                                color= mstm.ValidateArgs.check_color(
                                    x_axis["xaxis_axes_add_ref_color"], color_name="#999"
                                ),
                                linestyle=x_axis["xaxis_axes_add_ref_pattern"],
                                linewidth=x_axis["xaxis_axes_add_ref_width"],
                            )
                    except:
                        pass

            else:
                xticks([])


    def _customize_y_axes(self):
        """Customize y axis"""
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tick_params.html
        y, y_axis = self.y, self.y_axis

        if y_axis is not None:
            if y_axis["yaxis_show_labels"]:
                yaxis_title = mstm.titles_break(title_str=y_axis["yaxis_title"])
                ylabel(
                    ylabel=yaxis_title,
                    labelpad=y_axis["yaxis_title_topmargin"],
                    fontdict=y_axis["labels_font_properties"],
                    loc=y_axis["yaxis_title_location"],
                    verticalalignment=y_axis["yaxis_title_valign"],
                )

                tick_params(
                    axis=y_axis["yaxis_labels_axis"],  # 'x', 'y', 'both'
                    which="major",  #'major', 'minor', 'both'
                    reset=False,
                    direction=y_axis["yaxis_labels_tick_rules"],  # 'in', 'out', 'inout'
                    length=y_axis["yaxis_labels_ticks_length"],
                    width=y_axis["yaxis_labels_ticks_width"],
                    pad=y_axis["yaxis_labels_margin"],
                    labelsize=y_axis["yaxis_labels_font_size"],
                    colors= mstm.ValidateArgs.check_color(y_axis["yaxis_labels_ticks_color"]),
                    labelcolor= mstm.ValidateArgs.check_color(y_axis["yaxis_labels_font_color"]),
                    top=y_axis["yaxis_labels_top"],
                    labeltop=y_axis[
                        "yaxis_labels_top"
                    ],
                    labelrotation=y_axis["yaxis_labels_angle"],
                )

                # customize ticks
                if y_axis["yaxis_labels_ticks_customize"].lower() != "default":
                    #'Min, max and interval', 'Number of ticks', 'Custom'
                    a, b = asfarray([min(y), max(y)])
                    decimal_points = y_axis["yaxis_labels_decimals"]
                    ticks_customize_str = y_axis["yaxis_labels_ticks_customize"].lower()
                    if ticks_customize_str == "min and max":
                        ticks_values = round(asfarray([a, b]), decimal_points)
                        ticks_labels = ticks_values
                    elif ticks_customize_str == "min, max and interval":
                        try:
                            h = float(sympify(y_axis["yaxis_labels_ticks_value"]))
                            if len(h) != 1:
                                raise TypeError(f"Input must be a numeric value.")
                        except:
                            h = (b - a - 1) / 10
                        ticks_values = round(arr_abrange(a, b, h), decimal_points)
                        ticks_labels = ticks_values
                    elif ticks_customize_str == "number of ticks":
                        try:
                            n = int(sympify(y_axis["yaxis_labels_ticks_value"]))
                        except:
                            n = 10
                        ticks_values = round(
                            linspace(a, b, n), decimal_points
                        )
                        ticks_labels = ticks_values
                    elif ticks_customize_str == "custom":
                        values_labels = Matrix(
                            sympify(y_axis["yaxis_labels_ticks_value"])
                        )
                        ticks_values = flatten(values_labels[:, 0])
                        ticks_labels = flatten(values_labels[:, 1])

                    # add the ticks
                    yticks(ticks_values, ticks_labels)
                    try:
                        yticks(ticks_values, ticks_labels)
                    except:
                        pass

                # reference lines
                if y_axis["yaxis_axes_add_ref"]:
                    ref_points = flatten(
                        Matrix(sympify(y_axis["yaxis_axes_add_ref"]))
                    )
                    for point in ref_points:
                        axhline(
                            y=point,
                            color= mstm.ValidateArgs.check_color(
                                y_axis["yaxis_axes_add_ref_color"], color_name="#999"
                            ),
                            linestyle=y_axis["yaxis_axes_add_ref_pattern"],
                            linewidth=y_axis["yaxis_axes_add_ref_width"],
                        )
                    try:
                        ref_points = flatten(
                            Matrix(sympify(y_axis["yaxis_axes_add_ref"]))
                        )
                        for point in ref_points:
                            axhline(
                                y=point,
                                color= mstm.ValidateArgs.check_color(
                                    y_axis["yaxis_axes_add_ref_color"], color_name="#999"
                                ),
                                linestyle=y_axis["yaxis_axes_add_ref_pattern"],
                                linewidth=y_axis["yaxis_axes_add_ref_width"],
                            )
                    except:
                        pass

            else:
                yticks([])


    def _add_title(self):
        """Add figure title"""
        figure_title = self.figure_title

        titles_title_name = mstm.titles_break(title_str=figure_title["titles_title_name"])
        if figure_title["titles_subtitle_name"]:
            titles_title_name = (
                f"{titles_title_name}\n{figure_title['titles_subtitle_name']}"
            )
        title(
            label=titles_title_name,
            fontdict=figure_title["titles_title_properties"],
            loc=figure_title["titles_title_location"],
            pad=figure_title["titles_title_margin"],
            ha=figure_title["titles_title_halign"],
            va=figure_title["titles_title_valign"],
        )


    def _add_caption(self):
        """Add caption"""
        figure_caption = self.figure_caption
        x, y = self.x, self.y
        titles_caption_title = mstm.titles_break(title_str=figure_caption["titles_caption_title"])
        if isinstance(figure_caption["titles_caption_coordinates"], ndarray):
            xx, yy = figure_caption["titles_caption_coordinates"]
        else:
            xx, yy = min(x), min(y)

        text(
            x=xx,
            y=yy - float(figure_caption["titles_caption_topmargin"]),
            s=titles_caption_title,
            fontdict=figure_caption["titles_caption_properties"],
            ha=figure_caption["titles_caption_halign"],
            va=figure_caption["titles_caption_valign"],
        )


    def _add_note(self):
        """Add note"""
        x, y, figure_note = self.x, self.y, self.figure_note

        titles_note_title = mstm.titles_break(title_str=figure_note["titles_note_title"])
        if isinstance(figure_note["titles_note_coordinates"], ndarray):
            xx, yy = figure_note["titles_note_coordinates"]
        else:
            xx, yy = max(x), min(y)

        text(
            x=xx,
            y=yy - float(figure_note["titles_note_topmargin"]),
            s=titles_note_title,
            fontdict=figure_note["titles_note_properties"],
            ha=figure_note["titles_note_halign"],
            va=figure_note["titles_note_valign"],
        )

    def _add_legend(self):
        """Add legend"""
        figure_legend = self.figure_legend

        if not figure_legend["legend_hide"]:
            legend(
                loc=figure_legend["legend_location"],
                ncol=figure_legend["legend_columns"],
                markerfirst=figure_legend["legend_symbol_first"],
                frameon=figure_legend["legend_frameon"],
                framealpha=figure_legend["legend_frame_alpha"],
                borderpad=figure_legend["legend_border_pad"],
                facecolor= mstm.ValidateArgs.check_color(
                    figure_legend["legend_background_color"], color_name="#eee"
                ),
                edgecolor= mstm.ValidateArgs.check_color(
                    figure_legend["legend_edgecolor"], color_name="#eee"
                ),
                mode=figure_legend["legend_span_across"],
                labelcolor= mstm.ValidateArgs.check_color(
                    figure_legend["legend_label_font_color"], color_name="#333"
                ),
                prop={
                    "family": figure_legend["legend_label_font_name"],
                    "weight": figure_legend["legend_label_font_weight"],
                    "style": figure_legend["legend_label_font_style"],
                    "size": figure_legend["legend_label_font_size"],
                },
                title=figure_legend["legend_title_name"],
                title_fontproperties={
                    "family": figure_legend["legend_title_font_name"],
                    "weight": figure_legend["legend_title_font_weight"],
                    "style": figure_legend["legend_title_font_style"],
                    "size": figure_legend["legend_title_font_size"],
                },
            )


    def _customize_others(self):
        """Customize other graph properties"""
        figure_others = self.figure_others

        axis(figure_others["others_axes_appearance"])
        if figure_others["others_major_grid"]:
            grid(b=True, which="major", color="skyblue", linestyle="dotted")
        if figure_others["others_minor_grid"]:
            if figure_others["others_major_grid"]:
                grid(visible=True, which="major", color="skyblue", linestyle="solid")
                grid(visible=True, which="minor", color="grey", linestyle="dotted")
            else:
                grid(visible=True, which="major", axis="both", color="skyblue", linestyle="solid")
                grid(visible=True, which="minor", axis="both", color="skyblue", linestyle="dotted")
            minorticks_on()
        
        # to add others, let if be kwargs instead of others_...


    def generate_html(self):
        """Generate and return the HTML code for the figure"""
        figure_IO = BytesIO()
        savefig(figure_IO, format="png")
        figure_IO.seek(0)
        figure_decoded = base64.b64encode(figure_IO.getvalue()).decode("ascii")
        html_code = (
            f'<div style="text-align:center;"><img src="data:image/png;base64,'
            f'{figure_decoded}"></div>'
        )

        return html_code


def plot_styles() -> list:
    """
    Generate a list of available plotting styles for selection.

    Returns
    -------
    dropdown : list
        A list containing available plotting styles formatted for a dropdown 
        menu.
    """
    # Generate dropdown list of available plotting styles
    dropdown = ["Default"] + style.available[4:]  # Start from the fifth style
    dropdown = [["Plotting style", mstm.options_tuple(dropdown)]]

    return dropdown


def plot_scale() -> list:
    """
    Generate a list of available plot scale options for selection.

    Returns
    -------
    dropdown : list
        A list containing available plot scale options formatted for 
        a dropdown menu.
    """
    # Generate dropdown list of available plot scale options
    dropdown = ["default", "semilogx", "semilogy", "loglog"]
    dropdown = [["Legend position", mstm.options_tuple(dropdown)]]

    return dropdown


def legend_position() -> list:
    """
    Generate a list of available legend position options for selection.

    Returns
    -------
    dropdown : list
        A list containing available legend position options formatted for 
        a dropdown menu.
    """
    # Generate dropdown list of available legend position options
    dropdown = [
        "best",
        "upper right",
        "upper left",
        "lower left",
        "lower right",
        "right",
        "center left",
        "center right",
        "lower center",
        "upper center",
        "center",
    ]
    dropdown = [["Legend position", mstm.options_tuple(dropdown)]]

    return dropdown


def line_pattern() -> list:
    """
    Generate a list of available line pattern options for selection.

    Returns
    -------
    dropdown : list
        A list containing available line pattern options formatted for 
        a dropdown menu.
    """
    dropdown = ["none", "solid", "dashed", "dotted", "dashdot"]
    dropdown = [["Line pattern", mstm.options_tuple(dropdown)]]

    return dropdown


def marker_symbols() -> list:
    """
    Generate a list of available marker symbol options for selection.

    Returns
    -------
    dropdown : list
        A list containing available marker symbol options formatted 
        for a dropdown menu.
    """
    dropdown = [
        ("none", "None"),
        (".", "Point"),
        (",", "Pixel"),
        ("o", "Circle"),
        ("v", "Down triangle"),
        ("^", "Up triangle"),
        ("<", "Left triangle"),
        (">", "Right triangle"),
        ("1", "Tri down"),
        ("2", "Tri up"),
        ("3", "Tri left"),
        ("4", "Tri right"),
        ("s", "Square"),
        ("p", "Pentagon"),
        ("*", "Five star"),
        ("h", "Hexagon1"),
        ("H", "Hexagon2"),
        ("+", "Plus"),
        ("x", "Cross"),
        ("D", "Diamond"),
        ("d", "Thin diamond"),
        ("|", "vline"),
        ("_", "hline"),
    ]
    dropdown = [["Marker symbol", dropdown]]

    return dropdown


def h_align() -> list:
    """
    Generate a list of available horizontal alignment options for 
    selection.

    Returns
    -------
    dropdown : list
        A list containing available horizontal alignment options 
        formatted for a dropdown menu.
    """
    dropdown = ["center", "right", "left"]
    dropdown = mstm.options_tuple(dropdown)

    return dropdown


def v_align() -> list:
    """
    Generate a list of available vertical alignment options for 
    selection.

    Returns
    -------
    dropdown : list
        A list containing available vertical alignment options 
        formatted for a dropdown menu.
    """
    dropdown = ["center", "top", "bottom", "baseline", "center_baseline"]
    dropdown = mstm.options_tuple(dropdown)

    return dropdown


def yv_align() -> list:
    """
    Generate a list of available vertical alignment options for the y-axis 
    label.

    Returns
    -------
    dropdown : list
        A list containing available vertical alignment options formatted for 
        a dropdown menu.
    """
    dropdown = ["center", "top", "bottom"]
    dropdown = mstm.options_tuple(dropdown)

    return dropdown


def font_name():
    """
    Generate a list of available font name options for selection.

    Returns
    -------
    dropdown : list
        A list containing available font name options formatted for 
        a dropdown menu.
    """
    dropdown = ["consolas", "sans", "courier", "helvetica"]
    dropdown = mstm.options_tuple(dropdown)

    return dropdown


def font_weight():
    """
    Generate a list of available font weight options for selection.

    Returns
    -------
    dropdown : list
        A list containing available font weight options formatted for 
        a dropdown menu.
    """

    dropdown = [
        "ultralight",
        "light",
        "normal",
        "regular",
        "book",
        "medium",
        "roman",
        "semibold",
        "demibold",
        "demi",
        "bold",
        "heavy",
        "extra bold",
        "black",
    ]
    dropdown = mstm.options_tuple(dropdown)

    return dropdown


def font_style() -> list:
    """
    Generate a list of available font style options for selection.

    Returns
    -------
    dropdown : list
        A list containing available font style options formatted for 
        a dropdown menu.
    """
    dropdown = ["normal", "italic", "oblique"]
    dropdown = mstm.options_tuple(dropdown)

    return dropdown


def figure_layout() -> list:
    """
    Generate a list of available figure layout options for selection.

    Returns
    -------
    dropdown : list
        A list containing available figure layout options formatted for 
        a dropdown menu.
    """
    dropdown = ["on", "off", "equal", "scaled", "tight", "auto", "image", "square"]
    dropdown = mstm.options_tuple(dropdown)

    return dropdown


def location() -> list:
    """
    Generate a list of available location options for placing objects 
    or text within a plot.

    Returns
    -------
    dropdown : list
        A list containing available location options formatted for 
        a dropdown menu.
    """
    dropdown = ["center", "right", "left"]
    dropdown = mstm.options_tuple(dropdown)

    return dropdown


def box_styles() -> list:
    """
    Generate a list of available box styles for selection.

    Returns
    -------
    dropdown : list
        A list containing available box styles formatted for 
        a dropdown menu.
    """
    dropdown = [
        "none",
        "square",
        "circle",
        "larrow",
        "rarrow",
        "darrow",
        "round",
        "round4",
        "sawtooth",
        "roundtooth",
    ]
    dropdown = mstm.options_tuple(dropdown)

    return dropdown


def tick_rules():
    """
    Generate a list of available tick placement options for plots.

    Returns
    -------
    dropdown : list
        A list containing available tick placement options formatted for 
        a dropdown menu.
    """
    dropdown = [("in", "Inside"), ("out", "Outside"), ("inout", "Cross axis")]
    dropdown = [["Select placement", dropdown]]

    return dropdown


def ticks_customize():
    """
    Generate a list of options for customizing ticks in a plot.

    Returns
    -------
    dropdown : list
        A list containing options for customizing ticks formatted for 
        a dropdown menu.
    """
    dropdown = [
        "Default",
        "Min and max",
        "Min, max and interval",
        "Number of ticks",
        "Custom",
    ]
    dropdown = [["Select rule", mstm.options_tuple(dropdown)]]

    return dropdown


def labels_axis(x=True):
    """
    Generate a list of options for selecting the axis for label 
    customization.

    Parameters
    ----------
    x : bool, optional (default=True)
        Specifies whether the label customization is for the x-axis 
        (True) or y-axis (False).

    Returns
    -------
    dropdown : list
        A list containing options for selecting the axis formatted for 
        a dropdown menu.
    """
    if x:
        dropdown = [("x", "x axis")]
    else:
        dropdown = [("y", "y axis")]

    dropdown = [["Select axis", dropdown]]

    return dropdown


def labels_type():
    """
    Generate a list of options for selecting the type of labels to 
    customize on the axes.

    Returns
    -------
    dropdown : list
        A list containing options for selecting the type of labels 
        formatted for a dropdown menu.
    """
    dropdown = ["major", "minor", "both"]
    dropdown = mstm.options_tuple(dropdown, change_case=True)

    return dropdown
