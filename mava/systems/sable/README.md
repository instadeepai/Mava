# Sable

Sable is an algorithm that was developed by the research team at InstaDeep. It also casts MARL as a sequence modelling problem and leverages the [advantage decompostion theorem](https://arxiv.org/pdf/2108.08612) through auto-regressive action selection for convergence guarantees and can scale to thousands of agents by leveraging the memory efficiency of Retentive Networks.

We provide two Anakin based implementations of Sable:
* [ff-sable](../../systems/sable/anakin/ff_sable.py)
* [rec-sable](../../systems/sable/anakin/rec_sable.py)

Here the `ff` suffix implies that the algorithm retains no memory over time but treats only the agents as the sequence dimension while `rec` implies that the algorithms maintains memory over both agents and time for long context memory in partially observable environments.

For an overview of how the algorithm works, please see the diagram below. For a more detailed overview please see our associated [paper](https://arxiv.org/pdf/2410.01706).

<p align="center">
    <a href="../../../docs/images/algo_images/sable-arch.png">
        <img src="../../../docs/images/algo_images/sable-arch.png" alt="Sable Arch" width="80%"/>
    </a>
</p>

*Sable architecture and execution.* The encoder receives all agent observations $o_t^1,\dots,o_t^N$ from the current timestep $t$ along with a hidden state $h\_{t-1}^{\text{enc}}$ representing past timesteps and produces encoded observations $\hat{o}\_t^1,\dots,\hat{o}\_t^N$, observation-values $v \left( \hat{o}\_t^1 \right),\dots,v \left( \hat{o}\_t^N  \right) $, and a new hidden state $h_t^{\text{enc}}$.
The decoder performs recurrent retention over the current action $a_t^{m-1}$, followed by cross attention with the encoded observations, producing the next action $a_t^m$. The initial hidden states for recurrence over agents in the decoder at the current timestep are $( h\_{t-1}^{\text{dec}\_1},h\_{t-1}^{\text{dec}\_2})$, and by the end of the decoding process, it generates the updated hidden states $(h_t^{\text{dec}_1},h_t^{\text{dec}_2})$.

## Relevant paper:
* [Performant, Memory Efficient and Scalable Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/2410.01706)
* [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/pdf/2307.08621)
