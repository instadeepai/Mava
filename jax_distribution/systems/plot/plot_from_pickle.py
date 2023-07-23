import pickle 
import matplotlib.pyplot as plt
import numpy as np 

data_path = 'ppo_anakin_cleaner_results.pkl'
with open(data_path, 'rb') as f:
    data = pickle.load(f)

# output['metrics']['returned_episode_returns']

data = data.tolist()
data = np.array(data)
plot_data = np.mean(data, axis=(0, 2, 4)).reshape(-1)
plt.plot(plot_data)
plt.savefig('cleaner_results.png')