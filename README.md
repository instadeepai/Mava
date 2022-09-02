<p align="center">
    <a href="docs/images/mava.png">
        <img src="docs/images/mava.png" alt="Mava logo" width="70%"/>
    </a>
</p>

<h2 align="center">
    <p>A research framework for distributed multi-agent reinforcement learning in JAX</p>
</h2>

<div align="center">
<a rel="nofollow">
    <img src="https://img.shields.io/pypi/pyversions/id-mava" alt="Python" />
</a>
<a rel="nofollow">
    <img src="https://badge.fury.io/py/id-mava.svg" alt="PyPi" />
</a>
<a rel="nofollow">
    <img src="https://github.com/instadeepai/Mava/workflows/format_and_test/badge.svg" alt="Formatting" />
</a>
<a rel="nofollow">
    <img src="https://img.shields.io/lgtm/grade/python/g/instadeepai/Mava.svg?logo=lgtm&logoWidth=18" alt="Quality" />
</a>
<a rel="nofollow">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License" />
</a>
<a rel="nofollow">
    <img src="https://readthedocs.org/projects/id-mava/badge/?version=latest" alt="Docs" />
</a>
</div>

<p align="center">
  <img align="center" src="docs/images/animation_quick.gif" width="70%">
</p>

## Welcome to the Mava! 🦁

[**Installation**](#installation-)
| [**Quickstart**](#quickstart-)
| [**Documentation**](https://id-mava.readthedocs.io/)

Mava is a library for building multi-agent reinforcement learning (MARL) systems. Mava provides useful components, abstractions, utilities and tools for MARL and allows for simple scaling for multi-process system training and execution while providing a high level of flexibility and composability. Originating in the Research Team at [InstaDeep](https://www.instadeep.com/), Mava is now developed jointly with the open-source community. “Mava” means experience, or wisdom, in Xhosa - one of South Africa’s eleven official languages.

To join us in these efforts, please feel free to reach out, raise issues or read our [contribution guidelines](#contributing-) (or just star 🌟 to stay up to date with the latest developments)!

<hr>

👋 **UPDATE - 02/09/2022**: As mentioned in our previous update, we are in the process of fully moving to JAX. In the next few weeks, we will release our first JAX system! Using a more modular and flexible build method, with many more develops to come soon. These will include more components and utilities, state-of-the-art system implementations, notebooks and tutorials as well as detailed benchmarking and performance guides. Please note that all TF2-based systems using the old build method will be deprecated in the near future. As part of this update, we have also restructured the readme and our documentation to reflect our move to JAX. 

<hr>

### Overview 🦜

- 🥑 **Modular building blocks for MARL**: modular abstractions and [components](https://id-mava.readthedocs.io/en/latest/components/components/) for MARL to facilitate building multi-agent systems at scale.
- 🍬 **Environment Wrappers**: easily connect to your favourite MARL environment including [SMAC][smac], [PettingZoo][pettingzoo], [Flatland][flatland], [2D RoboCup][robocup], [OpenSpiel][openspiel] and more. For details on how to add your own environment, see [here](https://github.com/instadeepai/Mava/blob/develop/mava/wrappers/README.md).
- 🎓 **Educational Material**: [examples] and [user guides][quickstart] to facilitate Mava's adoption and highlight the added value of JAX-based MARL.

## Installation 🎬

You can install the latest release of Mava from PyPI:
```bash
pip install id-mava[reverb,jax,envs]
```
or you can install the latest development version directly from GitHub (recommended for latest JAX code):
```bash
pip install "id-mava[reverb,jax,envs] @ git+https://github.com/instadeepai/mava.git"
```

We have tested `mava` on Python 3.7, 3.8 and 3.9. Note that because the installation of JAX differs depending on your hardware accelerator,
we advise users to explicitly install the correct JAX version (see the [official installation guide](https://github.com/google/jax#installation)). For more in-depth instalations guides including Docker builds and virtual environments, please see our [optional installs](OPTIONAL_INSTALL.md).

## Quickstart ⚡

We have a [Quickstart notebook][quickstart] that can be used to quickly create and train your first Multi-Agent System. For more information on how to use Mava, please view our [usage section](#usage). Finally, for more on Mava's implementation details, please visit our [documentation].

## Contributing 🤝

Please read our [contributing docs](./CONTRIBUTING.md) for details on how to submit pull requests, our Contributor License Agreement and community guidelines.

## Troubleshooting and FAQs

Please read our [troubleshooting and FAQs guide](./TROUBLESHOOTING.md).

## Citing Mava

If you use Mava in your work, please cite the accompanying
[technical report][Paper]:

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
[openspiel]: https://github.com/deepmind/open_spiel
[meltingpot]: https://github.com/deepmind/meltingpot
[flatland]: https://gitlab.aicrowd.com/flatland/flatland
[robocup]: https://github.com/rcsoccersim
[pymarl]: https://github.com/oxwhirl/pymarl
[launchpad]: https://github.com/deepmind/launchpad
[quickstart]: https://github.com/instadeepai/Mava/blob/develop/examples/jax/quickstart.ipynb
[documentation]: https://id-mava.readthedocs.io/
