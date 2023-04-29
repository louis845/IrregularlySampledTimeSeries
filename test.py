import torch
import torch.distributions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
std_normal = torch.distributions.MultivariateNormal(loc=torch.zeros(3, device=device), covariance_matrix=torch.diag(torch.ones(3, device=device)))

print(std_normal.rsample((10,)).shape)