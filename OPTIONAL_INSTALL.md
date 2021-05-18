### Optional Installs
1. **Optional**: To install ROM files for Atari-Py using AutoROM (https://github.com/PettingZoo-Team/AutoROM).
   ```
   pip install autorom && AutoROM
   ```
   Then follow the on-screen instructions.

   You might also need to download unrar:
   ```
   sudo apt-get install unrar
   ```


2. **Optional**: To install opencv for [Supersuit](https://github.com/PettingZoo-Team/SuperSuit) environment wrappers.
    ```
    sudo apt-get update
    sudo apt-get install ffmpeg libsm6 libxext6  -y
    ```

3. **Optional**: To install [CUDA toolkit](https://docs.nvidia.com/cuda/) for NVIDIA GPU support, download [here](https://anaconda.org/anaconda/cudatoolkit). Alternatively, for anaconda users:

    ```bash
    conda install -c anaconda cudatoolkit
    ```

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
