import numpy as np
import matplotlib.pyplot as plt

def gaussian(n, m):
    return np.random.normal(0, 1, size=(n, m))

res = gaussian(100, 2)

plt.title(f"Gaussian Scatter Plot")
plt.scatter(res[: , 0], res[: , 1])
plt.show()