import numpy as np
import pandas as pd

matrix = np.array([
    [0, 0, 0.5, 0],
    [0, 0, 0, 1],
    [1.0, 0, 0, 0],
    [0, 1.0, 0.5, 0]
                  ])

# Solve r = 0.8 matrix @ r + 0.2 using fixed point iteration.
r = np.array([0.0, 0.0, 0.0, 0.0])

history = [r]
while True:
    r = 0.8 * matrix @ r + 0.2
    history.append(r)

    # stopping criterion, if the difference between the last two iterations is small enough
    if len(history) > 1 and np.max(np.abs(history[-1] - history[-2])) < 1e-5:
        break

print(pd.DataFrame(data = np.array(history), columns = ["P", "Q", "R", "S"]))