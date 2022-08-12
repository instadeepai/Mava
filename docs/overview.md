# Getting started with Mava

> 🚧 **Note:** This only applies to the callback redesign of Mava.

Going from papers to to code is one of the main challenges faced by reinformcement learning researchers. Mava aims to remove this constraint and to be the to go-to framework for fast iteration and experimentation. Mava also handles distributed computation by leveraging [Launcpad](https://github.com/deepmind/launchpad).

### Design Paradigm

Mava makes use of the a callback-based approach to enable modularity and flexibility in system designs. This enables the development of new system with minimal code reuse and overhead leading to fast experimentation and results. TODO (docs): GIVE BRIEF CALLBACK OVERVIEW // UPDATE LINK. More information on callbacks may be found [here](https://google.com). Hooks are overwritten in callback classes which are referred to as system components.

### Deep learning framework

Mava utilizes [Jax](https://github.com/google/jax) due to its accelator agnostic nature and computational speed gains due its ability to run on TPUs easily and just-in-time compilation.
