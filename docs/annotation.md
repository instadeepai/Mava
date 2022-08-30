# Annotation

> 🚧 **Note:** This only applies to the callback redesign of Mava.

It is common for MARL algorithms to have slightly different names across multiple studies.  In order to avoid this confusion in MAVA, we want to clarify some annotation conventions that we used in building our systems.

## IPPO vs. MAPPO

The two main learning schemes in MARL are **Independent learning** (decentralized critic) and **Centralized training, decentralized execution** (centralized-critic or some mixing network). In the multi-agent case, the PPO algorithm can be implemented for both schemes. We therefore use **IPPO** for the independent learning implementation of PPO and **MAPPO** for centralized training.

> ⚠️ **Note:** Currently, only the JAX system *IPPO* exists in the library, with plans to release *MAPPO* in the near future.

## Convention

We anticipate that more algorithms will be implemented using the redesigned JAX system in the near future, and we will try to adhere to the following convention:

- In the case of **independent learning** systems, we will annotate the algorithm using **"I"** (independent) to define the algorithm (e.g. **I**PPO).
- In the case of **centralized training** systems, we will use **"MA"** (multi-agent) to define the algorithm (e.g. **MA**PPO).
