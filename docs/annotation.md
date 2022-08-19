# Annotation

> 🚧 **Note:** This only applies to the callback redesign of Mava.

In MARL, it is usual to have several algorithms go by slightly different names across multiple studies.  So, in order to avoid this confusion in MAVA, we want to clarify some annotation conventions that we used in building our different systems.

## IPPO vs. MAPPO

The two main learning schemes in MARL are **The independent learning** (decentralized critic) and **The centralized training decentralized execution** (centralized-critic or some mixing network). In the multi-agent case, the PPO algorithm can be implemented for both cases.

As a regime, we used **IPPO** for the independent learning implementation of PPO and **MAPPO** for the centralized training.

> ⚠️ **Note:** This convention is still not followed yet, because the jax system *MAPPO* that we have in the library is currently the independent learning algorithm (Play the role of *IPPO*), and we are currently working on releasing the centralized algorithm using PPO, and once that is completed, we will change the names to conform with the convention.

## Convention

We anticipate that more algorithms will be implemented using the redesigned JAX system in the near future, and we will try to adhere to the following convention:

- In the case of **the independent learning** systems, we will annotate the algorithm using **"I"** (independent) to define the algorithm (e.g. **I**PPO).
- In the case of **the centralized training** systems, we will use **"MA"** (multi-agent) to define the algorithm (e.g. **MA**PPO).
