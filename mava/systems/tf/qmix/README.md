# QMIX (Q-value function factorisation)

An implementaiton of the QMIX MARL system ([Rashid et al., 2018]). QMIX is based on the idea of factorising the joint Q-value function for a team of agents and learning the weightings for each component using a monotonic mixing network whose weights are itself learned using a hypernetwork. 🔺 NOTE: our current implementation of QMIX has been not able to reproduce results demonstrated in the original paper.

<p style="text-align:center;">
<img src="https://raw.githubusercontent.com/instadeepai/Mava/develop/docs/images/qmix.png" width="80%">
</p>

[Rashid et al., 2018]: https://arxiv.org/pdf/1803.11485
