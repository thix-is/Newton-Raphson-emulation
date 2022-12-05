import time
import numpy as np
import pandas as pd
import scipy.stats as sp
from scipy import optimize

def call_option(S, r, vol, K, T):
    d1 = (np.log(S/K) + (r+0.5*vol**2)*T) / (vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    N1 = sp.norm.cdf(d1)
    N2 = sp.norm.cdf(d2)
    call = S*N1 - K*np.exp(-r*T)*N2
    return call

def call_vega(S, r, vol, K, T):
    d1 = (np.log(S/K) + (r+0.5*vol**2)*T) / (vol*np.sqrt(T))
    n1 = sp.norm.pdf(d1)
    vega = S*n1*np.sqrt(T)
    return vega

def vomma_zero(S, r, K, T):
    z = np.sqrt(np.abs(2*(np.log(S/K)+r*T)/T))
    return z


np.random.seed(0)
n=int(1e+6) # number of options
S=np.array(1.).astype(np.float32); r = np.array(0.).astype(np.float32)
v = np.random.uniform(0.01,0.5,n).astype(np.float32)
T = np.random.uniform(0.01,2,n).astype(np.float32)
lnK = np.random.uniform((r-0.5*v**2)*T-2*v*np.sqrt(T),
                        (r-0.5*v**2)*T+2*v*np.sqrt(T))
K = np.exp(lnK).astype(np.float32)


call = call_option(S,r,v,K,T)
f = lambda vol : call_option(S,r,vol,K,T) - call
df = lambda vol : call_vega(S,r,vol,K,T)
x0 = vomma_zero(S,r,K,T)


def counter():
    optime = []
    for i in range(100):
        start_time = time.process_time()
        imvol = optimize.newton(f, x0, fprime=df, maxiter=8)
        end_time = time.process_time()
        optime.append(int(round((end_time - start_time)*1000)))
    return optime, imvol
  

op, imvol = counter()
error = np.subtract(v,imvol)
mae = np.abs(error).mean()
mse = np.square(error).mean()
error2 = np.subtract(v,imvol) / v
mre = np.abs(error2).mean()
df = pd.DataFrame(op, columns=['Times'])


print('MAE : ',mae)
print('-'*30)
print('MSE : ',mse)
print('-'*30)
print('MRE : ',mre)
print('-'*30)
print(df.Times.describe())
