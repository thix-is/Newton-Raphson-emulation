import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from torch.distributions.normal import Normal

norm = Normal(0., 1.)
N = lambda x : norm.cdf(x)

def call_option(S,r,vol,K,T):
    
    d1 = (torch.log(S/K) + (r+0.5*vol**2)*T) / (vol*torch.sqrt(T))
    d2 = d1 - vol*torch.sqrt(T)
    N1 = N(d1)
    N2 = N(d2)
    call = S*N1 - K*torch.exp(-r*T)*N2
    
    return call

def call_vega(S,r,vol,K,T):
    
    d1 = (torch.log(S/K) + (r+0.5*vol**2)*T) / (vol*torch.sqrt(T))
    n1 = norm.log_prob(d1).exp()
    vega = S*n1*torch.sqrt(T)
    
    return vega

def vomma_zero(S,r,K,T):
    
    z = torch.sqrt(torch.abs(2*(torch.log(S/K)+r*T)/T))
    
    return z


    
class NRnet(nn.Module):
    
    def __init__(self, layers_num):
        
        super(NRnet,self).__init__()
        
        self.model = nn.ModuleList([])
        
        for layer in range(layers_num):
                   
            temp = nn.Linear(1,1,bias=False)
            temp.weight  = nn.Parameter(torch.ones([1,1]))
            self.model.append(temp)
            
    def forward(self, x):
        
        x0,x1,x2,x3,x4,x5 = torch.chunk(data,6,1)
        
        for i,layer in enumerate(self.model) :
        
            y = (call_option(x2,x3,x0,x4,x5) - x1)/call_vega(x2,x3,x0,x4,x5)
            x0 = x0 - layer(y)
            
        return x0
    
    
if __name__ == "__main__" :
    
    import onnx
    import onnxruntime as ort
    import os

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

    inputs = vomma_zero(S,r,K,T)
    
    
    layers_num = 8; rank = 0
    net = NRnet(layers_num)
    net = net.cuda()
    net = net.to(torch.float32)
    
    py_batch = net(data)
    
#     for i,p in enumerate(net.parameters()):

#         print(p)
    
    mse  = torch.mean( (v - py_batch)**2 ) 
    print(mse)

    # onnx
    input_names = ['data'] #input name
    output_names = ['imvol'] #output name

    torch.onnx.export(net,data,'NRlayers.onnx',verbose=False,input_names=input_names,output_names=output_names)
    onnx_model = onnx.load('NRlayers.onnx')
    onnx.checker.check_model(onnx_model) 
    
    # tensorrt
    import tensorrt as trt
    
    onnx_file_name = 'NRlayers.onnx'
    tensorrt_file_name = 'NRlayers.engine' #tensor rt engine name
    
    fp16_mode = False
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    
    #onnx to tensor rt engine
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network :

        parser = trt.OnnxParser(network, TRT_LOGGER)
        builder.max_workspace_size = (1 << 30) #1GB
        builder.fp16_mode = fp16_mode

        with open(onnx_file_name, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print (parser.get_error(error))

        #print("network.num_layers", network.num_layers)

        #network.mark_output(network.get_layer(network.num_layers - 1).get_output(0))
        engine = builder.build_cuda_engine(network)

        # buf = engine.serialize()
        # print(buf)
        with open(tensorrt_file_name, 'wb') as f:
            f.write(engine.serialize())
            
    onnx_model_ = onnx.load('NRlayers.onnx')
    onnx.checker.check_model( onnx_model_ )

    ort_session = ort.InferenceSession('NRlayers.onnx')
    py_batch = ort_session.run(None,{'data':data.detach().cpu().numpy()})
    py_batch = torch.as_tensor(py_batch, device='cuda:%d'%0 );
    py_batch.squeeze_(0)

    mse  = torch.mean( (v - py_batch)**2 ) 

    print( torch.cat( [py_batch[:10],v[:10]], axis=1 ) )
    print('(onnx) mse:%e'%(mse))

    
