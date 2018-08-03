import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.02

steps = np.linspace(0, np.pi * 2, 100, dtype=np.float32)
x_np = np.sin(steps)
y_np = np.cos(steps)
plt.plot(steps, y_np, 'r-', label='target(cos)')
plt.plot(steps, x_np, 'b-', label='input(sin)')
plt.legend(loc='best')
plt.show()
