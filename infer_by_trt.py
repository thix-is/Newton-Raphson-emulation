import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import os
import time
import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from torch.distributions.normal import Normal

def call_option(S,r,vol,K,T):
    
    d1 = (torch.log(S/K) + (r+0.5*vol**2)*T) / (vol*torch.sqrt(T))
    d2 = d1 - vol*torch.sqrt(T)
    N1 = Normal(0., 1.).cdf(d1)
    N2 = Normal(0., 1.).cdf(d2)
    call = S*N1 - K*torch.exp(-r*T)*N2
    
    return call

def call_vega(S,r,vol,K,T):
    
    d1 = (torch.log(S/K) + (r+0.5*vol**2)*T) / (vol*torch.sqrt(T))
    n1 = Normal(0., 1.).log_prob(d1).exp()
    vega = S*n1*torch.sqrt(T)
    
    return vega

def vomma_zero(S,r,K,T):
    
    z = torch.sqrt(torch.abs(2*(torch.log(S/K)+r*T)/T))
    
    return z
    
def allocate(engine,context) :
    
    h_input = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)
    
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    return (h_input,h_output,d_input,d_output)


dtype = torch.float
np.random.seed(0)
n=int(1e+6)
S= torch.ones((n,1), dtype=dtype).cuda()
r = torch.zeros((n,1), dtype=dtype).cuda()
v_np = np.random.uniform(0.01,0.5,n).reshape(n,1)
T_np = np.random.uniform(0.01,2,n).reshape(n,1)
lnK = np.random.uniform((-0.5*v_np**2)*T_np-2*v_np*np.sqrt(T_np),
                        (-0.5*v_np**2)*T_np+2*v_np*np.sqrt(T_np))
K_np = np.exp(lnK).reshape(n,1)

v = torch.from_numpy(v_np).float()
T = torch.from_numpy(T_np).float()
K = torch.from_numpy(K_np).float()

v = v.cuda(); T = T.cuda(); K = K.cuda()


call = call_option(S,r,v,K,T)
inputs = vomma_zero(S,r,K,T)
data = torch.cat([inputs,call,S,r,K,T],axis=1)
    
f = lambda vol : call_option(S,r,vol,K,T) - call
df = lambda vol : call_vega(S,r,vol,K,T)
    
inputs = vomma_zero(S,r,K,T).cpu().numpy() 
data = data.cpu().numpy()
v = v.cpu().numpy()


tensorrt_file_name_0 = 'NRlayers.engine'

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
result = []
for z in range(101): 
    
    with open(tensorrt_file_name_0,'rb') as f_0, trt.Runtime(TRT_LOGGER) as runtime_0:

        engine_0 = runtime_0.deserialize_cuda_engine(f_0.read())

        with engine_0.create_execution_context() as context_0:

            h_input_0,h_output_0,d_input_0,d_output_0 = allocate(engine_0,context_0) 

            np.copyto(h_input_0,data.flatten()) 

            stream = cuda.Stream()

            cuda.memcpy_htod(d_input_0, h_input_0)
            
            start = time.process_time()
            context_0.execute_v2(bindings=[int(d_input_0), int(d_output_0)])
            end = time.process_time()
            cost_time1 = (end-start)*1000
            cuda.memcpy_dtoh(h_output_0, d_output_0)

            py_batch = np.empty(n)
            np.copyto(py_batch,h_output_0)
            py_batch = py_batch.reshape([-1,1])

            loss = np.mean((py_batch-v)**2)
            mae  = np.mean(abs(py_batch-v))
            
            print(z)
            print('time : ', cost_time1)
            print('MSE : ', loss)
            print('MAE : ', mae)
            
            result.append(cost_time1)

result_df = pd.DataFrame(result)
result_df.to_csv('result_time.csv', index=False)

import pickle

with open('trt_imvol.pkl', 'wb') as f:
    pickle.dump(py_batch, f)