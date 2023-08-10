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
| [**Documentation**](https://id-mava.readthedocs.io/)

Mava is first and foremost a tool for quickly iterating on multi-agent reinforcement learning (MARL) ideas. Mava provides useful implementations of MARL algorithms that in JAX allowing for easy parallelisation across devices with JAX's `pmap`. Originating in the Research Team at [InstaDeep](https://www.instadeep.com/), Mava is now developed jointly with the open-source community. “Mava” means experience, or wisdom, in Xhosa - one of South Africa’s eleven official languages.

To join us in these efforts, please feel free to reach out, raise issues or read our [contribution guidelines](#contributing-) (or just star 🌟 to stay up to date with the latest developments)!

<hr>

👋 **UPDATE - 11/8/2023**: This is out first release of an end-to-end JAX version on Mava. Henceforth we will only be supporting JAX-based environments and systems with native support for the [Jumanji][jumanji] environment API. If you would still like to use our deprecated TF2-based systems please install [`v0.1.3`](https://github.com/instadeepai/Mava/releases/tag/0.1.3) of Mava (e.g. `pip install id-mava==0.1.3`).

<hr>

### Overview 🦜

- 🥑 **Implementations of MARL algorithms**: Implementations of multi-agent PPO systems that follow both the Centralised Training with Decentralised Execution (CTDE) and Decentralised Training with Decentralised Execution (DTDE) MARL paradigms. There are
- 🍬 **Environment Wrappers**: Example for mapping a Jumanji environment to an environment usable by Mava.
- 🎓 **Educational Material**:[user guides][quickstart] to facilitate Mava's adoption and highlight the added value of JAX-based MARL.

## Installation 🎬

You can install the latest release of Mava as follows:

```bash
pip install id-mava[reverb,jax,envs]
```

You can also install directly from source:

```bash
pip install "id-mava[reverb,jax,envs] @ git+https://github.com/instadeepai/mava.git"
```

We have tested `mava` on Python 3.8, 3.9 and 3.10. Note that because the installation of JAX differs depending on your hardware accelerator,
we advise users to explicitly install the correct JAX version (see the [official installation guide](https://github.com/google/jax#installation)). For more in-depth instalations guides including Docker builds and virtual environments, please see our [detailed installation guide](DETAILED_INSTALL.md).

## Quickstart ⚡

We have a [Quickstart notebook][quickstart] that can be used to quickly create and train your first Multi-Agent System. For more on Mava's implementation details, please visit our [documentation].

## Contributing 🤝

Please read our [contributing docs](./CONTRIBUTING.md) for details on how to submit pull requests, our Contributor License Agreement and community guidelines.

## Troubleshooting and FAQs

Please read our [troubleshooting and FAQs guide](./TROUBLESHOOTING.md).

## Performance



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

[Examples]: examples
[Paper]: https://arxiv.org/pdf/2107.01460.pdf
[pettingzoo]: https://github.com/PettingZoo-Team/PettingZoo
[smac]: https://github.com/oxwhirl/smac
[flatland]: https://gitlab.aicrowd.com/flatland/flatland
[quickstart]: https://github.com/instadeepai/Mava/blob/develop/examples/quickstart.ipynb
[documentation]: https://id-mava.readthedocs.io/
[jumanji]: https://github.com/instadeepai/jumanji
