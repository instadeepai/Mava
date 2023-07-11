import matplotlib.pyplot as plt
import numpy as np

# Simulate some data
environments = [16*4, 160*4, 1000*4, 5000*4, 10000*4, 20000*4]
gpu_times = [9.47, 15.14, 43.41, 182.29, 381.03]
tpu_times = [23.46, 28.06, 50.04, 206.65, 331.43, 590.17]

# Create a figure
fig, ax = plt.subplots()

# Plot data
print(environments[:len(gpu_times)])
ax.plot(environments[:len(gpu_times)], gpu_times, label='GPU')
ax.plot(environments[:len(tpu_times)], tpu_times, label='TPU', color='orange')

# Set labels and title
ax.set_xlabel('Number of Environments')
ax.set_ylabel('Time (Seconds)')
ax.set_title('GPU vs TPU')

# Add a legend
ax.legend()

# Show the plot
plt.show()

