# Detailed installation guide

### Docker

#### Using pre-built images

You can pull & run the latest pre-built images from our [DockerHub](https://hub.docker.com/r/instadeepct/mava) by specifying the docker image and example/file you want to run.

For example, this will pull the latest mava jax core image and run the `examples/debugging/simple_spread/feedforward/decentralised/run_ippo.py` example:

```
docker run --gpus all -it --rm  -v $(pwd):/home/app/mava -w /home/app/mava instadeepct/mava:jax-core-latest python examples/debugging/simple_spread/feedforward/decentralised/run_ippo.py --base_dir /home/app/mava/logs/
```

* For windows, replace `$(pwd)` with `$(curdir)`.

* You can replace the example with your custom python file.

#### Building the image yourself

1. Build the correct docker image using the `make` command:

    For Windows, before the docker image build, we recommend to first install the package manager [chocolatey](https://chocolatey.org/install) and run (to install make):

    ```bash
    choco install make
    ```

    1.1 Only Mava core:

    Jax version:

    ```bash
    make build version=jax-core
    ```

    1.2 For **optional** environments:

    * PettingZoo:

        ```
        make build version=pz
        ```

    * SMAC: The StarCraft Multi-Agent Challenge Environments :

        Install StarCraft II using a bash script, which is a slightly modified version of the script found [here][pymarl]:

        ```
        ./bash_scripts/install_sc2.sh
        ```

        Build Image

        ```
        make build version=sc2
        ```

    * Flatland:

        ```
        make build version=flatland
        ```

    * 2D RoboCup environment

        ```
        make build version=robocup
        ```

    To allow for agent recordings, where agents evaluations are recorded and these recordings are stored in a `/recordings` folder:

    ```
    make build version=[] record=true
    ```

2. Run an example:

    ```bash
    make run example=dir/to/example/example.py
    ```

    For example, `make run example=examples/petting_zoo/sisl/multiwalker/feedforward/decentralised/run_mad4pg.py`.

    Alternatively, run bash inside a docker container with mava installed, `make bash`, and from there examples can be run as follows: `python dir/to/example/example.py`.

    To run an example with tensorboard viewing enabled, you can run

    ```bash
    make run-tensorboard example=dir/to/example/example.py
    ```

    and navigate to `http://127.0.0.1:6006/`.

    To run an example where agents are recorded (**ensure you built the image with `record=true`**):

    ```
    make run-record example=dir/to/example/example.py
    ```

    Where example, is an example with recording available e.g. `examples/debugging/simple_spread/feedforward/decentralised/run_maddpg_record.py`.

### Python virtual environment

1. If not using docker, we strongly recommend using a
    [Python virtual environment](https://docs.python.org/3/tutorial/venv.html)
    to manage your dependencies in order to avoid version conflicts. Please note that since Launchpad only supports Linux based OSes, using a python virtual environment will only work in these cases:

    ```bash
    python3 -m venv mava
    source mava/bin/activate
    pip install --upgrade pip setuptools
    ```

    1.1  To install the core libraries, including [Reverb](https://github.com/deepmind/reverb) - our storage dataset , Tensorflow and [Launchpad](https://github.com/deepmind/launchpad) - for distributed agent support :

    * Install swig for box2d:

    ```bash
    sudo apt-get install swig -y
    ```

    * Install core dependencies:

    ```bash
    pip install id-mava[jax,reverb]
    ```

    * Or for the latest version of mava from source (**you can do this for all pip install commands below for the latest depedencies**):

    ```bash
    pip install git+https://github.com/instadeepai/Mava#egg=id-mava[reverb,jax]
    ```

    **If you are using `jax` and `CUDA`, you also need to run the following:**

    ```bash
    pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    ```

    1.2 For **optional** environments:
    * PettingZoo:

        ```
        pip install id-mava[pz]
        ```

    * Flatland:

        ```
        pip install id-mava[flatland]
        ```

    * 2D RoboCup environment:

        A local install has only been tested using the Ubuntu 18.04 operating system.
        The installation can be performed by running the RoboCup bash script while inside the Mava
        python virtual environment.

        ```bash
        ./bash_scripts/install_robocup.sh
        ```

    * StarCraft II:

        First install StarCraft II

        ```bash
        ./bash_scripts/install_sc2.sh
        ```

        Then set SC2PATH to the location of 3rdparty/StarCraftII, e.g. :

        ```
        export SC2PATH="/home/Documents/Code/Mava/3rdparty/StarCraftII"
        ```

        Then install the environment wrapper

        ```bash
        pip install id-mava[sc2]
        ```

3. Run an example:

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
