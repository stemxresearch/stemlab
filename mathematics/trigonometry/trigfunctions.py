from numpy import arccos, arcsin, arctan, deg2rad, cos, sin, tan
from stemlab.core.decimals import fround
from stemlab.core.arraylike import conv_to_arraylike
from stemlab.core.validators.validate import ValidateArgs


class TrigFunctions:
    """
    A class for trigonometric functions with support returned in
    degrees.

    Methods
    -------
    cosd()
        Cosine of `x`.
    sind()
        Sine of `x`.
    tand()
        Tangent of `x`.
    acosd()
        Arc sine of `x`.
    asind()
        Arc cosine of `x`.
    atand()
        Arc tangent of `x`.

    Examples
    --------
    >>> TrigFunctions.cosd(60)
    0.5
    >>> TrigFunctions.sind(30)
    0.5
    >>> TrigFunctions.tand(45)
    1.0
    >>> TrigFunctions.acosd(0.5)
    60.0
    >>> TrigFunctions.asind(0.5)
    30.0
    >>> TrigFunctions.atand(1)
    45.0
    """

    @staticmethod
    def _validate_args(x, is_inverse, decimal_points) -> None:
        """
        Validate the input arguments for trigonometric functions.

        Parameters
        ----------
        x : float
            The input value to validate.
        is_inverse : bool
            Whether the function is an inverse trigonometric function.
        decimal_points : int
            The number of decimal points to round the result to.

        Returns
        -------
        x : float
            The validated input value.
        decimal_points : int
            The validated number of decimal points.
        """
        limits = [-1, 1] if is_inverse else None
        try:
            float(x)
            x = ValidateArgs.check_numeric(par_name='x', limits=limits, user_input=x)
        except:
            x = conv_to_arraylike(
                array_values=x, to_ndarray=True, flatten_list=False, par_name='x'
            )
        decimal_points = ValidateArgs.check_decimals(x=decimal_points)
        return x, decimal_points

    @staticmethod
    def cosd(x: float, decimal_points: int = -1) -> float:
        """
        Returns the cosine of the angle `x` in degrees, rounded to `decimal_points`.

        Parameters
        ----------
        x : float
            The angle in degrees.
        decimal_points : int, optional (default=-1)
            The number of decimal points to round the result to.

        Returns
        -------
        x : {float, NumpyArray}
            The cosine of `x` rounded to `decimal_points`.

        Examples
        --------
        >>> TrigFunctions.cosd(60)
        0.5
        >>> TrigFunctions.cosd(90, decimal_points=2)
        0.0
        """
        x, decimal_points = TrigFunctions._validate_args(x=x, is_inverse=False, decimal_points=decimal_points)
        x = fround(cos(deg2rad(x)), to_Matrix=False, decimal_points=decimal_points)
        return x

    @staticmethod
    def sind(x: float, decimal_points: int = -1) -> float:
        """
        Returns the sine of the angle `x` in degrees, rounded to `decimal_points`.

        Parameters
        ----------
        x : float
            The angle in degrees.
        decimal_points : int, optional (default=-1)
            The number of decimal points to round the result to.

        Returns
        -------
        x : {float, NumpyArray}
            The sine of `x` rounded to `decimal_points`.

        Examples
        --------
        >>> TrigFunctions.sind(30)
        0.5
        >>> TrigFunctions.sind(45, decimal_points=3)
        0.707
        """
        x, decimal_points = TrigFunctions._validate_args(x=x, is_inverse=False, decimal_points=decimal_points)
        x = fround(sin(deg2rad(x)), to_Matrix=False, decimal_points=decimal_points)
        return x

    @staticmethod
    def tand(x: float, decimal_points: int = -1) -> float:
        """
        Returns the tangent of the angle `x` in degrees, rounded to `decimal_points`.

        Parameters
        ----------
        x : float
            The angle in degrees.
        decimal_points : int, optional (default=-1)
            The number of decimal points to round the result to.

        Returns
        -------
        x : {float, NumpyArray}
            The tangent of `x` rounded to `decimal_points`.

        Examples
        --------
        >>> TrigFunctions.tand(45)
        1.0
        >>> TrigFunctions.tand(60, decimal_points=2)
        1.732
        """
        x, decimal_points = TrigFunctions._validate_args(x=x, is_inverse=False, decimal_points=decimal_points)
        x = fround(tan(deg2rad(x)), to_Matrix=False, decimal_points=decimal_points)
        return x

    @staticmethod
    def acosd(x: float, decimal_points: int = -1) -> float:
        """
        Returns the arc cosine of `x` in degrees, rounded to `decimal_points`.

        Parameters
        ----------
        x : float
            The value for which to compute the arc cosine (must be in the range [-1, 1]).
        decimal_points : int, optional
            The number of decimal points to round the result to (default is -1 for no rounding).

        Returns
        -------
        float
            The arc cosine of `x` in degrees, rounded to `decimal_points`.

        Examples
        --------
        >>> TrigFunctions.acosd(0.5)
        60.0
        >>> TrigFunctions.acosd(0.707, decimal_points=2)
        45.0
        """
        x, decimal_points = TrigFunctions._validate_args(x=x, is_inverse=True, decimal_points=decimal_points)
        x = fround(deg2rad(arccos(x)), to_Matrix=False, decimal_points=decimal_points)
        return x

    @staticmethod
    def asind(x: float, decimal_points: int = -1) -> float:
        """
        Returns the arc sine of `x` in degrees, rounded to `decimal_points`.

        Parameters
        ----------
        x : float
            The value for which to compute the arc sine (must be in the range [-1, 1]).
        decimal_points : int, optional
            The number of decimal points to round the result to (default is -1 for no rounding).

        Returns
        -------
        float
            The arc sine of `x` in degrees, rounded to `decimal_points`.

        Examples
        --------
        >>> TrigFunctions.asind(0.5)
        30.0
        >>> TrigFunctions.asind(0.707, decimal_points=2)
        45.0
        """
        x, decimal_points = TrigFunctions._validate_args(x=x, is_inverse=True, decimal_points=decimal_points)
        x = fround(deg2rad(arcsin(x)), to_Matrix=False, decimal_points=decimal_points)
        return x

    @staticmethod
    def atand(x: float, decimal_points: int = -1) -> float:
        """
        Returns the arc tangent of `x` in degrees, rounded to `decimal_points`.

        Parameters
        ----------
        x : float
            The value for which to compute the arc tangent.
        decimal_points : int, optional
            The number of decimal points to round the result to (default is -1 for no rounding).

        Returns
        -------
        float
            The arc tangent of `x` in degrees, rounded to `decimal_points`.

        Examples
        --------
        >>> TrigFunctions.atand(1)
        45.0
        >>> TrigFunctions.atand(0.5, decimal_points=2)
        26.565
        """
        x, decimal_points = TrigFunctions._validate_args(x=x, is_inverse=True, decimal_points=decimal_points)
        x = fround(deg2rad(arctan(x)), to_Matrix=False, decimal_points=decimal_points)
        return x