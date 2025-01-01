## stemlab

A Python library for mathematical and Statistical Computing in Science, Technology, Engineering and Mathematics (STEM).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the stemlab library.

```bash
pip install stemlab
```

## Usage

The library is imported into a Python session by running the following import statement.

```python
>>> import stemlab as stm
>>> import numpy as np
```

## Mathematics examples

We will give a few examples that will demonstrate the use of the stemlab library.

### Richardson extrapolation

```python
>>> f = '2 ** x * sin(x)'
>>> x, n, h = (1.05, 4, 0.4)
>>> result = stm.diff_richardson(f, x, n, h, decimal_points=12)
```

### Gauss-Legendre quadrature integration

```python
>>> f = 'exp(-x) * cos(x)'
>>> a, b, n = (0, np.pi / 2, 6)
>>> result = stm.int_gauss_legendre(f, a, b, n, decimal_points=14)
```

### Fourth order Runge-Kutta method for solving IVPs

```python
>>> f = 'y - t^2 + 1'
>>> ft = '(t + 1)^2 - 0.5 * exp(t)'
>>> a, b = (0, 2)
>>> y0 = 0.5
>>> result = stm.ivps_rk4(odeqtn=f, exactsol=ft,
... time_span=[a, b], y0=y0, decimal_points=14)
```

### Newton divided interpolation

```python
>>> x = [1.0, 1.3, 1.6, 1.9, 2.2]
>>> y = [0.7651977, 0.6200860, 0.4554022, 0.2818186, 0.1103623]
>>> x0 = 1.5
>>> result = stm.interp_newton_divided(x, y, x0, diff_order=1,
... plot_x0=True, decimal_points=8)
```

## Statistics examples

```python
>>> import stemlab.statistical as sta
```

### Two independent samples t test using groups

```python
>>> df = stm.sta_dataset_read(name='scores')
>>> result = sta.means_2independent_t(sample1=df['score_after'],
... sample2_or_group=df['gender'], alternative='less', 
... sample_names = ['Female', 'Male'], decimal_points=4)
```

## Finance example

```python
>>> import stemlab.finance as stf
```

```python
>>> df = stf.amortization_schedule(principal=200000, annual_rate=6,
... years=1, schedule_type='monthly', decimal_points=2)
```

## Biology example

```python
>>> import pandas as pd
>>> import stemlab.biology as stb
>>> from stemlab.graphics import gph_scatter
```

```python
>>> N0 = 200
>>> r = 0.5
>>> K = 2000
>>> time = np.arange(1, 11)
>>> pop = []
>>> for t in time:
...     p = stb.logistic_growth(initial_pop=N0, rate=r, carrying_capacity=K, time=t)
...     pop.append(p)
>>> dframe = pd.DataFrame([time, pop], index=['Time', 'Population']).T
>>> display(dframe)
>>> gph_scatter(time, pop, xlabel='Time', ylabel='Population')
```

## Support

For any support on any of the functions in this library, send us an email at: ```library@stemxresearch.com```. We are willing to offer the necessary support where we possibly can.

## Roadmap

Future releases aim to make ```stemlab``` a first choice library for students, trainers and professionals in Science, Technology, Engineering and Mathematics (STEM).

## Contributing

To make stemlab a successful library while keeping the code easy. We welcome any valuable contributions towards the development and improvement of this library. 

For major changes to the library, please open an issue with us first to discuss what you would like to change and we will be more than willing to make the changes.

## Authors and acknowledgement

We are grateful to the incredible support from our developers at ```Stem Research```.

## License

[MIT](https://choosealicense.com/licenses/mit/)
