import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


cs = np.array([[16.66666667,  17.85714286,  19.23076923,  20.83333333,  23.80952381,   26.31578947,  25.,          31.25,        35.71428571,  33.33333333],
               [17.85714286,  20.,          19.23076923,  21.73913043,  22.72727273,
                   25.,   27.77777778,  33.33333333,  31.25,        45.45454545],
               [16.12903226,  19.23076923,  20.,          21.73913043,  23.80952381,
                   26.31578947,  31.25,        33.33333333,  35.71428571,  45.45454545],
               [17.24137931,  17.85714286,  20.83333333,  21.73913043,  23.80952381,
                   26.31578947,  31.25,        33.33333333,  38.46153846,  41.66666667],
               [18.51851852,  18.51851852,  20.,          22.72727273,  22.72727273,
                   27.77777778,  31.25,        31.25,        38.46153846,  50.],
               [17.24137931,  18.51851852,  21.73913043,  23.80952381,  25.,
                   27.77777778,   31.25,        35.71428571,  41.66666667,  50.],
               [16.66666667,  19.23076923,  18.51851852,  26.31578947,  25.,
                   31.25,   33.33333333,  35.71428571,  41.66666667,  50.],
               [16.12903226,  22.72727273,  21.73913043,  22.72727273,  25.,
                   33.33333333,   33.33333333,  35.71428571,  38.46153846,  38.4615384615],
               [17.85714286,  20.83333333,  21.73913043,  26.31578947,  26.31578947,
                   26.31578947,  31.25,        38.46153846,  45.45454545,  45.45454545],
               [17.85714286,  20.,          23.80952381,  23.80952381,  26.31578947,   29.41176471,  33.33333333,  33.33333333,  41.66666667,  45.45454545]])

cs_1 = np.zeros((25, 10))

for i in range(10):
    # given values
    xi = np.linspace(25, 75, 10)
    yi = cs[:, i]
# positions to inter/extrapolate
    x = np.linspace(10, 100, 25)
# spline order: 1 linear, 2 quadratic, 3 cubic ...
    order = 1
# do inter/extrapolation
    s = InterpolatedUnivariateSpline(xi, yi, k=order)
    y = s(x)
    cs_1[:, i] = y

print [cs_1]

# example showing the interpolation for linear, quadratic and cubic
# interpolation
