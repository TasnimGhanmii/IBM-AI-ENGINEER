import torch
import numpy as np
import pandas as pd

intTensor=torch.tensor([0,1,2,3,4])

#converting
floatTensor=torch.FloatTensor([0,1,2,3,4])

floatTensor=intTensor.type(torch.FloatTensor)

print("The size of the new_float_tensor: ", floatTensor.size())
print("The dimension of the new_float_tensor: ",floatTensor.ndimension())

#dim change
twoDTensor=intTensor.view(5,1)
#-1 means any size, you can only set 1 arg as -1!!!
twoDTensor=intTensor.view(-1,1)

#convert to numpy
#npArray and intTensor still point to np_array. As a result if we change numpy_array both back_to_numpy and new_tensor will change. 
np_array=intTensor.numpy()
#convert to tensor
np_array=np.array([0,1,2,3,4])
intTensor=np_array.from_nmpy(np_array)

#tensor fct
math_tensor = torch.tensor([1.0, -1.0, 1, -1])
print("Tensor example: ", math_tensor)

mean = math_tensor.mean()
print("The mean of math_tensor: ", mean)

standard_deviation = math_tensor.std()
print("The standard deviation of math_tensor: ", standard_deviation)

max_min_tensor = torch.tensor([1, 1, 3, 5, 5])
print("Tensor example: ", max_min_tensor)
max_val = max_min_tensor.max()
print("Maximum number in the tensor: ", max_val)
min_val = max_min_tensor.min()
print("Minimum number in the tensor: ", min_val)

pi_tensor = torch.tensor([0, np.pi/2, np.pi])
sin = torch.sin(pi_tensor)
print("The sin result of pi_tensor: ", sin)

len_5_tensor = torch.linspace(-2, 2, steps = 5)
print ("First Try on linspace", len_5_tensor)

#basic ops
u = torch.tensor([1, 0])
v = torch.tensor([0, 1])
w = u + v
print("The result tensor: ", w)
v = u + 1
print ("Addition Result: ", v)
u = torch.tensor([1, 2])
v = 2 * u
print("The result of 2 * u: ", v)
u = torch.tensor([1, 2])
v = torch.tensor([3, 2])
w = u * v
print ("The result of u * v", w)
print("Dot Product of u, v:", torch.dot(u,v))


