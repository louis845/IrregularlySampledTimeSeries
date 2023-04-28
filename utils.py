import torch
import torchdiffeq


def feedforward_nn(input_dim, output_dim, hidden_dim, num_hidden_layers, activation = torch.nn.Tanh, device = torch.device("cpu")):
    # Create a list of layers, starting with the input layer.
    layers = [torch.nn.Linear(input_dim, hidden_dim)]

    # Add num_hidden_layers hidden layers.
    for i in range(num_hidden_layers):
        layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        layers.append(activation())

    # Add the output layer.
    layers.append(torch.nn.Linear(hidden_dim, output_dim))

    # Create a sequential model from the layers.
    model = torch.nn.Sequential(*layers)

    # Move the model to the device.
    model.to(device)

    # Return the model.
    return model

class ODEFuncWrapper(torch.nn.Module):
    def __init__(self, ode_func: torch.nn.Module):
        """
        :param ode_func: The ODE function to use.
        """
        super(ODEFuncWrapper, self).__init__()
        self.ode_func = ode_func

    def forward(self, t, x):
        """
        :param t: The time step to evaluate the ODE function at.
        :param x: The input to the ODE function.
        :return: The output of the ODE function.
        """
        return self.ode_func(x)