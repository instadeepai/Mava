import json
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

path_to_results = "/home/ruanjohn/Work/Mava/results/43.json"

# Load the results
with open(path_to_results, "r") as f:
    out = json.load(f)

x = 0

ys = np.array(out["returns"]).mean(-1).reshape(-1)
xs = np.linspace(0, out["run_time"], len(ys))

plt.plot(xs, ys, label="VMAP 6GB GPU")
plt.xlabel("Seconds")
plt.ylabel("Return")
plt.savefig("results_time_new.png")
plt.close()
