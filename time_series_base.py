import torch
import abc



# create an abstract class called TimeSeriesBase that inherits from torch.nn.Module
class TimeSeriesBase(abc.ABC):
    def __init__(self, time_series_dims, device = torch.device("cpu")):
        """
        :param time_series_dims: The dimensions of the time series. Note that the initial state and the predictions are returned as a 2 * time_series_dims vector, to include the variance (confidence) of the predictions.
        :param ode_func: The ODE function to use
        :param device: The device to use for the model
        """
        self.device = device
        self.time_series_dims = time_series_dims

    @abc.abstractmethod
    def forward(self, initial_state, time_steps):
        """
        :param initial_state: The initial state of the time series. Should be a tensor of shape (..., 2 * self.time_series_dims).
        :param time_steps: The time steps to predict the time series on.
        :return: A tensor containing the predicted values (states). This would be a tensor of shape (..., len(time_steps), 2 * self.time_series_dims).
        """
        pass

