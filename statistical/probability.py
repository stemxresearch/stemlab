from typing import Literal


def standardize(x: float, mean: float, std: float):
    pass


def unstandardize(x: float, mean: float, std: float):
    pass


def normal_calculate_prob_given_x(
    x1: float,
    x2: float,
    mean: float,
    std: float,
    prob_type: Literal['less', 'greater', 'between']
) -> float:
    pass


def prob_normal_lt_x(
    x: float,
    mean: float,
    std: float,
    calculate_prob: bool = True
) -> float:
    pass
    

def prob_normal_gt_x(
    x1: float,
    x2: float,
    mean: float,
    std: float,
    calculate_prob: bool = True
) -> float:
    pass


def prob_normal_between_x1_x2(
    x1: float,
    x2: float,
    mean: float,
    std: float,
    calculate_prob: bool = True
) -> float:
    pass