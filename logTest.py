import numpy as np
import matplotlib.pyplot as plt

# Example data following an exponential curve
x = np.linspace(1, 10, 100)
y = 2 * np.exp(0.5 * x) + np.random.normal(0, 0.5, size=len(x))

# Logarithmic transformation
y_log = np.log(y)

# Plot original and transformed data
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(x, y, label='Original Data')
plt.title('Original Data')

plt.subplot(1, 2, 2)
plt.scatter(x, y_log, label='Log-Transformed Data')
plt.title('Log-Transformed Data')

plt.show()
