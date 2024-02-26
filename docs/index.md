<div style="text-align: center;">
    <a href="images/logos/mava_full_logo.png">
        <img src="images/logos/mava_full_logo.png" alt="Mava logo" style="width: 50%;"/>
    </a>
</div>

<h2 style="text-align: center;">
    <p>Distributed Multi-Agent Reinforcement Learning in JAX</p>
</h2>

<div style="text-align: center;">
<a href="https://www.python.org/doc/versions/">
      <img src="https://img.shields.io/badge/python-3.9-blue" alt="Python Versions">
</a>
<a  href="https://github.com/instadeepai/Mava/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-orange.svg" alt="License" />
</a>
<a  href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code Style" />
</a>
<a  href="https://mypy-lang.org/">
    <img src="https://www.mypy-lang.org/static/mypy_badge.svg" alt="MyPy" />
</a>
<a href="https://arxiv.org/pdf/2107.01460.pdf">
    <img src="https://img.shields.io/badge/PrePrint-ArXiv-red" alt="ArXiv">
</a>
<a href="https://colab.research.google.com/github/instadeepai/Mava/blob/develop/examples/Quickstart.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
</div>

## Welcome to Mava! 🦁

Mava provides simplified code for quickly iterating on ideas in multi-agent reinforcement learning (MARL) with useful implementations of MARL algorithms in JAX allowing for easy parallelisation across devices with JAX's `pmap`.
Mava is a project originating in the Research Team at [InstaDeep](https://www.instadeep.com/).

**If you are looking for documentation, benchmarks, contribution guidelines and more, you can find them [here](https://id-mava.readthedocs.io/)!**

To join us in these efforts, please feel free to reach out, raise issues or read our [contribution guidelines](#contributing) (or just star 🌟 to stay up to date with the latest developments)!

## Overview 🦜

Mava offers single-file implementations of Reinforcement Learning (RL) algorithms inspired by [PureJaxRL][purejaxrl]and [CleanRL][cleanrl].
These implementations are designed for clarity and ease of understanding while leveraging the advantages of JAX, such as `pmap` and `vmap`, to enhance efficiency and scalability in research and practice.

Mava currently offers the following building blocks for MARL research:

- **MARL Algorithms**: Implementations of multi-agent PPO systems that follow both the Centralised Training with Decentralised Execution (CTDE) and Decentralised Training with Decentralised Execution (DTDE) MARL paradigms.
- **Environment Wrappers**: Example wrappers for mapping Jumanji environments to an environment that is compatible with Mava. At the moment, we support [Robotic Warehouse][jumanji_rware] and [Level-Based Foraging][jumanji_lbf] with plans to support more environments soon. We have also recently added support for the SMAX environment from [JaxMARL][jaxmarl].
- **Educational Material**: [Quickstart notebook][quickstart] to demonstrate how Mava can be used and to highlight the added value of JAX-based MARL.
- **Statistically Robust Evaluation**: Mava natively supports logging to json files which adhere to the standard suggested by [Gorsane et al. (2022)][toward_standard_eval]. This enables easy downstream experiment plotting and aggregation using the tools found in the [MARL-eval][marl_eval] library.

## Getting Started 🎬

At the moment Mava is not meant to be installed as a library, but rather to be used as a research tool.

You can use Mava by cloning the repo and pip installing as follows:

```bash
git clone https://github.com/instadeepai/mava.git
cd mava
pip install -e .
```

We have tested `Mava` on Python 3.9. Note that because the installation of JAX differs depending on your hardware accelerator,
we advise users to explicitly install the correct JAX version (see the [official installation guide](https://github.com/google/jax#installation)). For more in-depth installation guides including Docker builds and virtual environments, please see our [detailed installation guide](install.md).

## Quickstart ⚡

To get started with training your first Mava system, simply run one of the system files. e.g.,

```bash
python mava/systems/ff_ippo.py
```

Mava makes use of Hydra for config management. In order to see our default system configs please see the `mava/configs/` directory. A benefit of Hydra is that configs can either be set in config yaml files or overwritten from the terminal on the fly. For an example of running a system on the LBF environment, the above code can simply be adapted as follows:

```bash
python mava/systems/ff_ippo.py env=lbf
```

Different scenarios can also be run by making the following config updates from the terminal:

```bash
python mava/systems/ff_ippo.py env=rware env/scenario=tiny-4ag
```

Additionally, we also have a [Quickstart notebook][quickstart] that can be used to quickly create and train your first Multi-agent system.

## Advanced Usage 👽

Mava can be used in a wide array of advanced systems. As an example, we demonstrate recording experience data from one of our PPO systems into a [Flashbax](https://github.com/instadeepai/flashbax) `Vault`. This vault can then easily be integrated into offline MARL systems, such as those found in [OG-MARL](https://github.com/instadeepai/og-marl). See the [Advanced README](other/advanced_usage.md) for more information.

## Contributing 🤝

Please read our [contributing docs](contributing.md) for details on how to submit pull requests, our Contributor License Agreement and community guidelines.

## Roadmap 🛤️

We plan to iteratively expand Mava in the following increments:

- 🌴 Support for more environments.
- 🔁 More robust recurrent systems.
- 🌳 Support for non JAX-based environments.
- 🦾 Support for off-policy algorithms.
- 🎛 Continuous action space environments and algorithms.

Please do follow along as we develop this next phase!

## TensorFlow 2 Mava ⚠️
Originally Mava was written in Tensorflow 2. Support for the TF2-based framework and systems has now been fully **deprecated**. If you would still like to use it, please install `v0.1.3` of Mava (i.e. `pip install id-mava==0.1.3`).

## See Also 🔎

**InstaDeep's MARL ecosystem in JAX.** In particular, we suggest users check out the following sister repositories:

- 🔌 [OG-MARL](https://github.com/instadeepai/og-marl): datasets with baselines for offline MARL in JAX.
- 🌴 [Jumanji](https://github.com/instadeepai/jumanji): a diverse suite of scalable reinforcement learning environments in JAX.
- 😎 [Matrax](https://github.com/instadeepai/matrax): a collection of matrix games in JAX.
- ⚡ [Flashbax](https://github.com/instadeepai/flashbax): accelerated replay buffers in JAX.
- 📈 [MARL-eval][marl_eval]: standardised experiment data aggregation and visualisation for MARL.

**Related.** Other libraries related to accelerated MARL in JAX.

- 🦊 [JaxMARL](https://github.com/flairox/jaxmarl): accelerated MARL environments with baselines in JAX.
- 🌀 [DeepMind Anakin][anakin_paper] for the Anakin podracer architecture to train RL agents at scale.
- ♟️ [Pgx](https://github.com/sotetsuk/pgx): JAX implementations of classic board games, such as Chess, Go and Shogi.
- 🔼 [Minimax](https://github.com/facebookresearch/minimax/): JAX implementations of autocurricula baselines for RL.

## Citing Mava 📚

If you use Mava in your work, please cite the accompanying
[technical report][Paper]:

```bibtex
@article{dekock2023mava,
    title={Mava: a research library for distributed multi-agent reinforcement learning in JAX},
    author={Ruan de Kock and Omayma Mahjoub and Sasha Abramowitz and Wiem Khlifi and Callum Rhys Tilbury
    and Claude Formanek and Andries P. Smit and Arnu Pretorius},
    year={2023},
    journal={arXiv preprint arXiv:2107.01460},
    url={https://arxiv.org/pdf/2107.01460.pdf},
}
```

## Acknowledgements 🙏

We would like to thank all the authors who contributed to the previous TF version of Mava: Kale-ab Tessera, St John Grimbly, Kevin Eloff, Siphelele Danisa, Lawrence Francis, Jonathan Shock, Herman Kamper, Willie Brink, Herman Engelbrecht, Alexandre Laterre, Karim Beguir. Their contributions can be found in our [TF technical report](https://arxiv.org/pdf/2107.01460v1.pdf).

The development of Mava was supported with Cloud TPUs from Google's [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC) 🌤.

[Paper]: https://arxiv.org/pdf/2107.01460.pdf
[quickstart]: https://github.com/instadeepai/Mava/blob/develop/examples/Quickstart.ipynb
[jumanji]: https://github.com/instadeepai/jumanji
[cleanrl]: https://github.com/vwxyzjn/cleanrl
[purejaxrl]: https://github.com/luchris429/purejaxrl
[jumanji_rware]: https://instadeepai.github.io/jumanji/environments/robot_warehouse/
[jumanji_lbf]: https://github.com/sash-a/jumanji/tree/feat/lbf-truncate
[epymarl]: https://github.com/uoe-agents/epymarl
[anakin_paper]: https://arxiv.org/abs/2104.06272
[rware]: https://github.com/semitable/robotic-warehouse
[jaxmarl]: https://github.com/flairox/jaxmarl
[toward_standard_eval]: https://arxiv.org/pdf/2209.10485.pdf
[marl_eval]: https://github.com/instadeepai/marl-eval
[smax]: https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/smax
