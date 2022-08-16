# Getting started with Mava

> 🚧 **Note:** This only applies to the callback redesign of Mava.

Going from papers to code is one of the main challenges faced by reinformcement learning researchers. Mava aims to remove this constraint and to be the go-to framework for fast iteration and experimentation. Mava also handles distributed computation by leveraging [Launchpad](https://github.com/deepmind/launchpad).

### Design Paradigm

Mava makes use of the a callback-based approach to enable modularity and flexibility in system designs. This enables the development of new system with minimal code reuse and overhead leading to fast experimentation and results. The system's architecture is based on components (Hooks) that are overwritten in callbacks, and these callbacks are passed as arguments to the system classes, allowing you to configure your system differently dependent on the algorithm's properties. More information on callbacks may be found [here](https://google.com). 

### Deep learning framework

Mava employs [Jax](https://github.com/google/jax) thanks to its accelator agnostic nature, computational speed gains, and its ability to run on TPUs easily and just-in-time compilation.
