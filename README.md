<p align="center">
    <a href="docs/images/new_logo_name.png">
        <img src="docs/images/new_logo_name.png" alt="Mava logo" width="50%"/>
    </a>
</p>

<h2 align="center">
    <p>A framework for distributed multi-agent reinforcement learning in JAX</p>
</h2>

<div align="center">
<a  href="https://pypi.org/project/id-mava/">
    <img src="https://img.shields.io/pypi/pyversions/id-mava" alt="Python" />
</a>
<a  href="https://pypi.org/project/id-mava/">
    <img src="https://badge.fury.io/py/id-mava.svg" alt="PyPi" />
</a>
<a  href="https://github.com/instadeepai/Mava/actions/workflows/ci.yaml?query=branch%3Adevelop">
    <img src="https://github.com/instadeepai/Mava/workflows/format_and_test/badge.svg" alt="Formatting" />
</a>
<a  href="https://github.com/instadeepai/Mava/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License" />
</a>
<a  href="https://id-mava.readthedocs.io/">
    <img src="https://readthedocs.org/projects/id-mava/badge/?version=latest" alt="Docs" />
</a>
<a href="https://codecov.io/gh/instadeepai/Mava">
    <img src="https://codecov.io/gh/instadeepai/Mava/branch/develop/graph/badge.svg?token=P3HO6DLKQ3" alt="coverage" />
</a>
</div>

## Welcome to Mava! 🦁

[**Installation**](#installation-)
| [**Quickstart**](#quickstart-)

Mava is first and foremost a tool for quickly iterating on multi-agent reinforcement learning (MARL) ideas. Mava provides useful implementations of MARL algorithms that in JAX allowing for easy parallelisation across devices with JAX's `pmap`. Originating in the Research Team at [InstaDeep](https://www.instadeep.com/), Mava is now developed jointly with the open-source community. “Mava” means experience, or wisdom, in Xhosa - one of South Africa’s eleven official languages.

To join us in these efforts, please feel free to reach out, raise issues or read our [contribution guidelines](#contributing-) (or just star 🌟 to stay up to date with the latest developments)!

<hr>

👋 **UPDATE - 11/8/2023**: This is out first release of an end-to-end JAX version on Mava. Henceforth we will only be supporting JAX-based environments and systems with native support for the [Jumanji][jumanji] environment API. The reason for this change is to have a lightweight and easy-to-use end-to-end JAX based research tool. We are currently following a similar design philosophy to [CleanRL][cleanrl] and [PureJaxRL][purejaxrl] where we allow for code duplication to enable readability. All algorithmic logic can be found in the file implementing a particular algorithm. If you would still like to use our deprecated TF2-based systems please install [`v0.1.3`](https://github.com/instadeepai/Mava/releases/tag/0.1.3) of Mava (e.g. `pip install id-mava==0.1.3`).

<hr>

### Overview 🦜

- 🥑 **Implementations of MARL algorithms**: Implementations of multi-agent PPO systems that follow both the Centralised Training with Decentralised Execution (CTDE) and Decentralised Training with Decentralised Execution (DTDE) MARL paradigms. There are
- 🍬 **Environment Wrappers**: Example wrapper for mapping a Jumanji environment to an environment usable by Mava. At the moment we only support [Robotic Warehouse][jumanji_rware] but plan to support more environments soon.
- 🎓 **Educational Material**:[user guides][quickstart] to facilitate Mava's adoption and highlight the added value of JAX-based MARL.

## Installation 🎬

At the moment Mava is not meant to be installed as a library, but rather to be used as a research tool.

You can use Mava by cloning the repo and pip installing as follows:

```bash
pip install -e .
```

We have tested `mava` on Python 3.9. Note that because the installation of JAX differs depending on your hardware accelerator,
we advise users to explicitly install the correct JAX version (see the [official installation guide](https://github.com/google/jax#installation)). For more in-depth instalations guides including Docker builds and virtual environments, please see our [detailed installation guide](DETAILED_INSTALL.md).

## Quickstart ⚡

We have a [Quickstart notebook][quickstart] that can be used to quickly create and train your first Multi-Agent System. For more on Mava's implementation details, please visit our [documentation].

## Contributing 🤝

Please read our [contributing docs](./CONTRIBUTING.md) for details on how to submit pull requests, our Contributor License Agreement and community guidelines.

## Performance and Speed
It should be noted that in all cases here Mava is trained using a single 6GB Nivida 3060 laptop GPU.

In order to show the utility of end-to-end JAX-based MARL systems and JAX-based environments we compare the speed and performance of Mava against EPyMARL on a simple Robotic Warehouse task with 4 agents.

Furthermore, we illustrate the speed of JAX-based MARL by illustrating the system steps per second as the number of parallel environments are increased

Here, also note the system performance on a larger set of Robotic Warehouse environments:
## Citing Mava

If you use Mava in your work, please cite the accompanying
[technical report][Paper] (to be updated soon to reflect our transition to JAX):

```bibtex
@article{pretorius2021mava,
    title={Mava: A Research Framework for Distributed Multi-Agent Reinforcement Learning},
    author={Arnu Pretorius and Kale-ab Tessera and Andries P. Smit and Kevin Eloff
    and Claude Formanek and St John Grimbly and Siphelele Danisa and Lawrence Francis
    and Jonathan Shock and Herman Kamper and Willie Brink and Herman Engelbrecht
    and Alexandre Laterre and Karim Beguir},
    year={2021},
    journal={arXiv preprint arXiv:2107.01460},
    url={https://arxiv.org/pdf/2107.01460.pdf},
}
```

## See Also 🔎

The current version of Mava has been based on code from the following projects:

- 🤖 [PureJaxRL][purejaxrl] provides simple code implementations for end-to-end RL training in JAX.
- 🌳 [EPyMARL][epymarl] provides a framework for training MARL systems using a PyTorch backend.
- 🦾 [DeepMind Anakin][anakin_notebook] provides a notebook that illustrates using the Anakin podracer architecture for training RL systems in JAX at scale.

[Examples]: examples
[Paper]: https://arxiv.org/pdf/2107.01460.pdf
[pettingzoo]: https://github.com/PettingZoo-Team/PettingZoo
[smac]: https://github.com/oxwhirl/smac
[flatland]: https://gitlab.aicrowd.com/flatland/flatland
[quickstart]: https://github.com/instadeepai/Mava/blob/develop/examples/quickstart.ipynb
[documentation]: https://id-mava.readthedocs.io/
[jumanji]: https://github.com/instadeepai/jumanji
[cleanrl]: https://github.com/vwxyzjn/cleanrl
[purejaxrl]: https://github.com/luchris429/purejaxrl
[jumanji_rware]: https://instadeepai.github.io/jumanji/environments/robot_warehouse/
[epymarl]: https://github.com/uoe-agents/epymarl
[anakin_notebook]: https://colab.research.google.com/drive/1974D-qP17fd5mLxy6QZv-ic4yxlPJp-G?usp=sharing
