from sympy import Matrix, sympify, latex

from stemlab.core.datatypes import ArrayMatrixLike


def latex_linear_systems(
    A: ArrayMatrixLike, 
    b: ArrayMatrixLike, 
    displaystyle: bool = True, 
    hspace: int = 0, 
    vspace: int = 7, 
    inline: bool = True, 
    auto_print: bool = True
):
    """
     Convert a linear system represented by matrices A and b into 
     LaTeX format.

    Parameters
    ----------
    A : array_like
        Coefficients matrix.
    b : array_like
        Constants matrix.
    displaystyle : bool, optional (default=True)
        Whether to use displaystyle in LaTeX.
    hspace : int, optional (default=0)
        Horizontal space in centimeters.
    vspace : int, optional (default=7)
        Vertical space in points.
    inline : bool, optional (default=True)
        Whether to use inline or display mode in LaTeX.
    auto_print : bool, optional (default=True)
        Whether to automatically print the LaTeX output.

    Returns
    -------
    Axb : str
        LaTeX representation of the linear system.
    """
    A = Matrix(A)  # .replace(0, 123456789.987654)
    b = Matrix(b)
    Ax = []
    # create LHS Ax
    for row in range(A.shape[0]):
        terms_joined = " + ".join(
            [f"{value} * x{k + 1}" for k, value in enumerate(A[row, :])]
        )
        Ax.append(sympify(terms_joined))
    Ax = Matrix(Ax)
    # join LHS (Ax) to RHS (b) to form Ax = b
    Axb = []
    for row in range(A.shape[0]):
        Axb.append([f"{latex(Ax[row, 0])} &= {latex(b[row, 0])}"])

    displaystyle_str = "\\displaystyle " if displaystyle else ""
    dollar = "$" if inline else "$$"
    hspace_str = "" if hspace else f"\\hspace{{{hspace}cm}}"
    delimiter = f" \\\\[{str(vspace)}pt] \n\t"
    Axb = delimiter.join(Axb)
    Axb = (
        f"{dollar}\n{hspace_str}{displaystyle_str}\n"
        f"\\begin{{aligned}}\n\t{Axb}\n\\end{{aligned}}\n{dollar}"
    )
    if auto_print:
        print(Axb)

    return Axb
