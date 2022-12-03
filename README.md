Newton-Raphson Emulation Network
================================
Introduction
------------
The implied volatility of the option can be obtained through the Newton-Raphson method of iterative method. But, iterative method is unsuitable for calculating numerous implied volatilities due to excessive computation. Therefore the Newton-Raphson(NR) method is implemented as a deep learning neural network to efficiently estimate a large amount of implied volatilities. The Newton-Raphson emulation network emulate the following Newton-Raphson method:   
$$\sigma  _{n+1} = \sigma  _{n} - {{h _{r,k,\tau } (\sigma  _{n} )-c _{mkt}} \over {h _{r,k,\tau }\prime (\sigma  _{n} )}}$$   

Numerical experiments
---------------------


Benchmarking
------------
The table below shows the result of comparing the model we proposed with Python ‘SciPy’ and ‘py_vollib_vectorized’ packages. The table shows the computation times (in milliseconds) for estimating the implied volatility. Each value is calculated by averaging the values from 100 repetitions, and the corresponding standard deviation is provided in parentheses.
<p align="center">
<img src="https://user-images.githubusercontent.com/119658929/205438877-9d31454a-9a25-41bd-ad8d-02c5233d364b.PNG"></p>
