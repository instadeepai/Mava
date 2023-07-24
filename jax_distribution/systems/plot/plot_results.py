import pickle

import matplotlib.pyplot as plt
import numpy as np

path_to_results = "/home/ruanjohn/Work/Mava/training_results_keep.pkl"

# Load the results
with open(path_to_results, "rb") as f:
    out = pickle.load(f)

ys = out["returned_episode_returns"].mean(-1).reshape(-1)
xs = np.linspace(0, 2074, len(ys))
xs2 = np.linspace(0, 9600, len(ys))

plt.plot(xs, ys, label="VMAP 6GB GPU")
plt.plot(xs2, ys, label="EpyMARL 40GB GPU")
plt.xlabel("Seconds")
plt.ylabel("Return")
plt.savefig("results_time.png")
plt.close()


plt.figure()
plt.plot(ys)
plt.xlabel("Update steps")
plt.ylabel("Return")
plt.savefig("results_updates.png")
