### Optional Installs
1. **Optional**: To install ROM files for Atari-Py using [AutoROM](https://github.com/PettingZoo-Team/AutoROM).
   ```
   pip install autorom && AutoROM
   ```
   Then follow the on-screen instructions.

   You might also need to download unrar-free:
   ```
   sudo apt-get install unrar-free
   ```


2. **Optional**: To install opencv for [Supersuit](https://github.com/PettingZoo-Team/SuperSuit) environment wrappers.
    ```
    sudo apt-get update
    sudo apt-get install ffmpeg libsm6 libxext6  -y
    ```

3. **Optional**: For GPU support follow NVIDIA's cuda download [instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) or download cudatoolkit through [anaconda](https://anaconda.org/anaconda/cudatoolkit).

4. **Optional**: To log episodes in video/gif format, using the `Monitor` wrapper.
- Install `xvfb` to run a headless/fake screen and `ffmpeg` to record video.
    ```
    sudo apt-get install -y xvfb ffmpeg
    ```

- Setup fake display:
    ```
    xvfb-run -s "-screen 0 1400x900x24" bash
    python [script.py]
    ```
    or
    ```
    xvfb-run -a python [script.py]
    ```

- Install `array2gif`, if you would like to save the episode in gif format.
    ```
    pip install .[record_episode]
    ```

- Note when rendering StarCraft II it may be necessary to point to the correct C++ version like this:
  ```bash
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/
  python examples/jax/smac/feedforward/decentralised/run_ippo.py
  ```

  or

  ```bash
  LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/ python examples/jax/smac/feedforward/decentralised/run_ippo.py
  ```
