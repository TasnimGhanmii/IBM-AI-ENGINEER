import torch
import numpy as np
import pandas as pd

tensor2D=torch.tensor([[11, 12, 13],[21, 22, 23],[31, 32, 33]])
print("The dimension of twoD_tensor: ", tensor2D.ndimension())
print("The shape of twoD_tensor: ", tensor2D.shape)
print("The shape of twoD_tensor: ", tensor2D.size())
print("The number of elements in twoD_tensor: ", tensor2D.numel())

#convert to numpy & vice versa
twoD_numpy = tensor.numpy()
print("Tensor -> Numpy Array:")
print("The numpy array after converting: ", twoD_numpy)
print("Type after converting: ", twoD_numpy.dtype)

print("================================================")

new_twoD_tensor = torch.from_numpy(twoD_numpy)
print("Numpy Array -> Tensor:")
print("The tensor after converting:", new_twoD_tensor)
print("Type after converting: ", new_twoD_tensor.dtype) 

#convert pandas series
df = pd.DataFrame({'A':[11, 33, 22],'B':[3, 3, 2]})
converted_tensor = torch.tensor(df.values)
print ("Tensor: ", converted_tensor)

#ops
X = torch.tensor([[1, 0],[0, 1]]) 
Y = torch.tensor([[2, 1],[1, 2]])
X_plus_Y = X + Y
print("The result of X + Y: ", X_plus_Y)

two_Y = 2 * Y
print("The result of 2Y: ", two_Y)

X_times_Y = X * Y
print("The result of X * Y: ", X_times_Y)

A = torch.tensor([[0, 1, 1], [1, 0, 1]])
B = torch.tensor([[1, 1], [1, 1], [-1, 1]])
A_times_B = torch.mm(A,B)
print("The result of A * B: ", A_times_B)
