import torch
import torchdiffeq
import numpy as np

"""grucell = torch.nn.GRUCell(2, 3, device=torch.device("cuda"))
output = grucell(torch.randn(2, 5, 2, device=torch.device("cuda")), torch.randn(2, 5, 3, device=torch.device("cuda")))
print(output.shape)"""

"""class TempModule(torch.nn.Module):
    def __init__(self):
        super(TempModule, self).__init__()

    def forward(self, t, x):
        print("t: ", t.shape, "    x: ", x.shape)
        return x

func = TempModule()

res = torchdiffeq.odeint_adjoint(func, torch.ones(3, 5), torch.linspace(0, 1, 10))
print(res)
print(res.shape)"""

z = np.random.normal(size=(20))
print(z)

for iter in range(len(z)-1):
    print(z[iter:(iter+2)])