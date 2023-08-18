# Detailed installation guide

### Docker

#### Building the image

1. Build the docker image using the `make` command:

    For Windows, before the docker image build, we recommend to first install the package manager [chocolatey](https://chocolatey.org/install) and run (to install make):

    ```bash
    choco install make
    ```

    To then build the docker image run

    ```bash
    make build
    ```

2. Run a system:

    ```bash
    make run example=dir/to/system.py
    ```

    For example, `make run example=mava/systems/ff_ippo_rware.py`.

    Alternatively, run bash inside a docker container with mava installed by running `make bash`, and from there systems can be run as follows: `python dir/to/system.py`.

### Python virtual environment

1. If not using docker, we strongly recommend using a
    [Python virtual environment](https://docs.python.org/3/tutorial/venv.html)
    to manage your dependencies in order to avoid version conflicts.

    ```bash
    python3 -m venv mava
    source mava/bin/activate
    pip install --upgrade pip setuptools
    git clone https://github.com/instadeepai/Mava.git
    cd mava
    ```

    1.1  To install the core libraries, please run:

    * Install core dependencies:

    ```bash
    pip install .
    ```

    **If you plan on using `JAX` along with a GPU or TPU:**

    Please follow the instruction on the official [`JAX`](https://github.com/google/jax) project repository.

3. Developing features for mava

    When developing features for MAVA one can replace the `id-mava` part of the `pip install` with a `.` (or mava's directory) in order to install all dependencies.

    ```bash
    git clone https://github.com/instadeepai/mava.git
    pip install -e "mava[reverb,jax]"
    ```

    This installs the mava dependencies into your current environment, but uses your local copy of mava, instead of the one stored inside of your environments package directory.

4. Run an example:

    ```
    python dir/to/example/example.py
    ```

    For certain examples and extra functionality such as the use of Atari environments, environment wrappers, gpu support and agent episode recording, we also have a list of [optional installs](OPTIONAL_INSTALL.md).

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

* Install `xvfb` to run a headless/fake screen and `ffmpeg` to record video.

    ```
    sudo apt-get install -y xvfb ffmpeg
    ```

* Setup fake display:

    ```
    xvfb-run -s "-screen 0 1400x900x24" bash
    python [script.py]
    ```

    or

    ```
    xvfb-run -a python [script.py]
    ```

* Install `array2gif`, if you would like to save the episode in gif format.

    ```
    pip install .[record_episode]
    ```

* Note when rendering StarCraft II it may be necessary to point to the correct C++ version like this:

  ```bash
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/
  python  examples/smac/feedforward/decentralised/run_ippo.py
  ```

  or

  ```bash
  LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/ python  examples/smac/feedforward/decentralised/run_ippo.py
  ```

[pymarl]: https://github.com/oxwhirl/pymarl
