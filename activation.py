import torch
import torch.nn.functional as F


#define the function hyperbolic tangent
def sigmoid_f(x):
  sigmoid = 1/(1 + torch.exp(-x))
  return sigmoid

# define the function derivate hyperbolic tangent
def d_sigmoid_f(x):
  d_sigmoid = sigmoid_f(x)*(1-sigmoid_f(x))
  return d_sigmoid


#define the function hyperbolic tangent
def tanh_f(x):
  tan_h = (torch.exp(x) - torch.exp(-x))/(torch.exp(x) + torch.exp(-x))
  return tan_h

# define the function derivate hyperbolic tangent
def dtanh(x):
  d_tan_h = 1 - torch.square(tanh_f(x))
  return d_tan_h


#define the function hyperbolic tangent
def reLU_f(x):
  relu = torch.tensor(torch.Tensor.where(x>0, x, 0))
  relu.cpu().numpy()
  # print(relu.dtype)
  return relu

# define the function derivate hyperbolic tangent
def d_reLU_f(x):
  d_relu = torch.max(0, 1)
  return d_relu


def swish(x):
  
  return x*sigmoid_f(x)

def d_swish(x):
  
  d = sigmoid_f(x) + x*sigmoid_f(x)*(1 - swish(x))
  return d



def softplus(x):
  return torch.log(1+torch.exp(x))

def mish(x):
 
  return x*tanh_f(softplus(x))

def d_mish(beta,x):
  
  d= mish(x)/x +x*sigmoid_f(x)*(1 - torch.square(tanh_f(softplus(x))))
  return d