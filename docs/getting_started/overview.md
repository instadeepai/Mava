# Getting started with Mava



Going from research papers to code and being able to quickly test experimental hypotheses are some of the main challenges faced by reinforcement learning researchers. Mava aims to remove this constraint and to be the go-to framework for fast iteration and experimentation. Mava also effectively utilizes computational resources by leveraging [Launchpad](https://github.com/deepmind/launchpad) for distributed computation.

### Design Paradigm

Mava makes use of the a callback-based approach to enable modularity and flexibility in system designs. This enables the development of new systems with maximal code reuse and minimal overhead leading to fast experimentation and results. The system's architecture is based on components that are overwritten in callbacks, and these callbacks are passed as arguments to the system classes, allowing users to configure systems differently depending on an implemented algorithm's properties. More information on callbacks may be found [here](../components/components.md).

### Deep learning framework

Mava employs [Jax](https://github.com/google/jax) thanks to its accelerator agnostic nature, computational speed gains from just-in-time compilation, and its ability to leverage the most recent deep learning hardware.
