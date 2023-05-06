"""Demo of glucose series inference using gradio packages and the pretrained encoder decoder models."""

import torch
import encoder_ode_rnn
import decoder
import utils
import time_series_sampler
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import io

def update_plot(noise, n_points, avg_sep, decay):
    print(noise)
    print(n_points)
    print(avg_sep)
    print(decay)

    # Generate time series
    x = np.linspace(0, 10, 1000)
    y = np.sin(x) + noise * np.random.randn(1000)
    # Sample time series
    indices = np.arange(0, len(x), avg_sep * (1 - decay))
    sample_x = x[indices[:n_points]]
    sample_y = y[indices[:n_points]]
    # Fit a polynomial to the sampled time series
    coeffs = np.polyfit(sample_x, sample_y, 3)
    poly = np.poly1d(coeffs)
    # Evaluate the polynomial over the full time series
    fit_y = poly(x)
    # Create a Matplotlib plot
    fig, ax = plt.subplots()
    ax.plot(x, y, label='Original')
    ax.plot(x, fit_y, label='Fit')
    ax.plot(sample_x, sample_y, 'o', label='Sampled')
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Sampled Time Series with Inference')
    ax.grid(True)
    # Save the plot to a file-like object
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    # Return the file-like object as Gradio output
    return buffer

iface = gr.Interface(
    update_plot,
    inputs=[
        gr.inputs.Slider(0.0, 1.0, step=0.01, default=0.0, label="Noise"),
        gr.inputs.Slider(2, 40, step=1, default=4, label="Number of Points"),
        gr.inputs.Slider(0.5, 2.0, step=0.01, default=1.0, label="Average Separation"),
        gr.inputs.Slider(0.3, 1.8, step=0.01, default=0.9, label="Decay"),
    ],
    outputs=gr.outputs.Image(type="numpy"),
    title="Sampled Time Series with Inference",
    description="This app generates a time series with random noise, samples it at regular intervals, "
                "tries to forecast using Latent ODE-RNN encoder decoder model, and shows the original time series, "
                "sampled points, and the predictions.",
    examples=[
        [0.0, 4, 1.0, 0.9]
    ],
    live=False,
)

iface.launch()