Newton-Raphson Emulation Network
================================
The implied volatility of the option can be obtained through the Newton-Raphson method of iterative method. But, iterative method is unsuitable for calculating numerous implied volatilities due to excessive computation. Therefore the Newton-Raphson(NR) method is implemented as a deep learning neural network to efficiently estimate a large amount of implied volatilities. The Newton-Raphson emulation network emulate the following Newton-Raphson method:   
$$\sigma  _{n+1} = \sigma  _{n} - {{h _{r,k,\tau } (\sigma  _{n} )-c _{mkt}} \over {h _{r,k,\tau }\prime (\sigma  _{n} )}}$$   

Numerical experiments
---------------------


Results
-------
