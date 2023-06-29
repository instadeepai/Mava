import pickle 
import matplotlib.pyplot as plt
import numpy as np 

path_to_results = "/home/ruanjohn/Work/Mava/training_results.pkl"

# Load the results
with open(path_to_results, 'rb') as f:
    out = pickle.load(f)

ys = out["returned_episode_returns"].mean(-1).reshape(-1)
xs = np.linspace(0, 2074, len(ys))

plt.plot(xs, ys)
plt.xlabel("Seconds")
plt.ylabel("Return")
plt.savefig('results_time.png')

plt.close()
plt.figure()
plt.plot(ys)
plt.xlabel("Update steps")
plt.ylabel("Return")
plt.savefig('results_updates.png')