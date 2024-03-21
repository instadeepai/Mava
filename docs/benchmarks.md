
## StarCraft Multi-Agent Challenge in JAX (SMAX) 👾
For comparing Mava’s stability to other JAX-based baseline algorithms, we train Mava’s recurrent IPPO and MAPPO systems on a broad range of [SMAX][smax] tasks.
We trained Mava’s recurrent systems on eight SMAX scenarios.
In all cases we do not rerun baselines but instead take results for final win rates from the [JaxMARL technical report](https://arxiv.org/pdf/2311.10090.pdf).
The outcomes were then compared to the final win rates reported by [Rutherford et al., 2023](https://arxiv.org/pdf/2311.10090.pdf).
To ensure fair comparisons, we also train Mava's system up to 10 million timesteps with 64 vectorised environments.


<p style="text-align:center;">
    <img src="../images/smax_results/legend.png" alt="legend" style="width: 55%;"/><br>
</p>

<div style="display: flex; justify-content: space-evenly; flex-wrap: wrap;">
    <div style="flex-basis: 30%; margin-bottom: 20px;">
        <img src="../images/smax_results/2s3z.png" alt="2s3z" style="width: 100%; max-width: 100%;">
        <div style="text-align:center;"><code>2s3z</code></div>
    </div>
    <div style="flex-basis: 30%; margin-bottom: 30px;">
        <img src="../images/smax_results/3s_vs_5z.png" alt="3s_vs_5z" style="width: 100%; max-width: 100%;">
        <div style="text-align:center;"><code>3s_vs_5z</code></div>
    </div>
    <div style="flex-basis: 30%; margin-bottom: 30px;">
        <img src="../images/smax_results/3s5z_vs_3s6z.png" alt="3s5z_vs_3s6z" style="width: 100%; max-width: 100%;">
        <div style="text-align:center;"><code>3s5z_vs_3s6z</code></div>
    </div>
    <div style="flex-basis: 30%; margin-bottom: 30px;">
        <img src="../images/smax_results/3s5z.png" alt="3s5z" style="width: 100%; max-width: 100%;">
        <div style="text-align:center;"><code>3s5z</code></div>
    </div>
    <div style="flex-basis: 30%; margin-bottom: 30px;">
        <img src="../images/smax_results/5m_vs_6m.png" alt="5m_vs_6m" style="width: 100%; max-width: 100%;">
        <div style="text-align:center;"><code>5m_vs_6m</code></div>
    </div>
    <div style="flex-basis: 30%; margin-bottom: 30px;">
        <img src="../images/smax_results/6h_vs_8z.png" alt="6h_vs_8z" style="width: 100%; max-width: 100%;">
        <div style="text-align:center;"><code>6h_vs_8z</code></div>
    </div>
    <div style="flex-basis: 30%; margin-bottom: 30px;">
        <img src="../images/smax_results/10m_vs_11m.png" alt="10m_vs_11m" style="width: 100%; max-width: 100%;">
        <div style="text-align:center;"><code>10m_vs_11m</code></div>
    </div>
    <div style="flex-basis: 30%; margin-bottom: 30px;">
        <img src="../images/smax_results/27m_vs_30m.png" alt="27m_vs_30m" style="width: 100%; max-width: 100%;">
        <div style="text-align:center;"><code>27m_vs_30m</code></div>
    </div>
</div>
<p style="font-style: italic; text-align: center; margin-top: 10px;"> Mava recurrent performance on SMAX tasks.</p>

## Robotic Warehouse 🤖

All experiments below were performed using an NVIDIA Quadro RTX 4000 GPU with 8GB Memory.

In order to show the utility of end-to-end JAX-based MARL systems and JAX-based environments we compare the speed of Mava against [EPyMARL][epymarl] as measured in total training wallclock time on simple [Robotic Warehouse][rware] (RWARE) tasks with 2 and 4 agents. Our aim is to illustrate the speed increases that are possible with using end-to-end Jax-based systems and we do not necessarily make an effort to achieve optimal performance. For EPyMARL, we use the hyperparameters as recommended by [Papoudakis et al. (2020)](https://arxiv.org/pdf/2006.07869.pdf) and for Mava we performed a basic grid search. In both cases, systems were trained up to 20 million total environment steps using 16 vectorised environments.






<p style="text-align:center;">
    <img src="../images/rware_results/ff_mappo/main_readme/legend.png" alt="legend" width="100%"/><br>
</p>
<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <div style="width: 30%; margin-right: 20px;">
        <img src="../images/rware_results/ff_mappo/main_readme/tiny-2ag-1.png" alt="Mava ff mappo tiny 2ag" style="width: 100%;">
        <div style="text-align:center;"><code>tiny-2ag</code></div>
    </div>
    <div style="width: 31%; margin-right: 20px;">
        <img src="../images/rware_results/ff_mappo/main_readme/tiny-4ag-1.png" alt="Mava ff mappo tiny 4ag" style="width: 100%;">
        <div style="text-align:center;"><code>tiny-4ag</code></div>
    </div>
    <div style="width: 31%; margin-right: 20px;">
        <img src="../images/rware_results/ff_mappo/main_readme/small-4ag-1.png" alt="Mava ff mappo small 4ag" style="width: 100%;">
        <div style="text-align:center;"><code>small-4ag</code></div>
    </div>
</div>
<p style="font-style: italic; text-align: center; margin-top: 10px;"> Mava feedforward MAPPO performance on RWARE tasks.</p>


**An important note on the differences in converged performance:**

In order to benefit from the wallclock speed-ups afforded by JAX-based systems it is required that environments also be written in JAX. It is for this reason that Mava does not use the exact same version of the RWARE environment as EPyMARL but instead uses a JAX-based implementation of RWARE found in [Jumanji][jumanji_rware], under the name RobotWarehouse. One of the notable differences in the underlying environment logic is that RobotWarehouse will not attempt to resolve agent collisions but will instead terminate an episode when agents do collide. In our experiments, this appeared to make the environment more challenging. For this reason we show the performance of Mava on Jumanji with and without termination upon collision indicated with `w/o collision` in the figure legends. For a more detailed discussion, please see the following [page](other/jumanji_rware_comparison.md).

## Level-Based Foraging 🌳
Mava also supports [Jumanji][jumanji_lbf]'s LBF. We evaluate Mava's recurrent MAPPO system on LBF, against [EPyMARL][epymarl] (we used original [LBF](https://github.com/semitable/lb-foraging) for EPyMARL) in 2 and 4 agent settings up to 20 million timesteps. Both systems were trained using 16 vectorized environments. For the EPyMARL systems we use an NVIDIA A100 GPU and for the Mava systems we use a GeForce RTX 3050 laptop GPU with 4GB of memory. To show how Mava can generalise to different hardware, we also train the Mava systems on a TPU v3-8. We plan to publish comprehensive performance benchmarks for all Mava's algorithms across various LBF scenarios soon.

<p style="text-align:center;">
    <img src="../images/lbf_results/legend_rec_mappo.png" alt="legend" width="70%"/><br>
</p>
<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <div style="width: 30%; margin-right: 20px;">
        <img src="../images/lbf_results/2s-8x8-2p-2f-coop_rec_mappo.png" alt="Mava rec mappo 2s-8x8-2p-2f-coop" style="width: 100%;">
        <div style="text-align:center;"><code>2s-8x8-2p-2f-coop</code></div>
    </div>
    <div style="width: 31%; margin-right: 20px;">
        <img src="../images/lbf_results/15x15-4p-3f_rec_mappo.png" alt="Mava rec mappo 15x15-4p-3f" style="width: 100%;">
        <div style="text-align:center;"><code>15x15-4p-3fz</code></div>
    </div>
</div>
<p style="font-style: italic; text-align: center; margin-top: 10px;"> Mava recurrent MAPPO performance on Level-based Foraging tasks.</p>


## Vectorised Environment Performance 🚀

Furthermore, we illustrate the speed of Mava by showing the steps per second as the number of parallel environments is increased. These steps per second scaling plots were computed using a standard laptop GPU, specifically an RTX-3060 GPU with 6GB memory.

<div style="text-align: center;">
    <a href="../images/speed_results/mava_sps_results.png">
        <img src="../images/speed_results/mava_sps_results.png" alt="Mava sps" style="width: 55%;"/>
    </a>
    <a href="../images/speed_results/ff_mappo_speed_comparison.png">
        <img src="../images/speed_results/ff_mappo_speed_comparison.png" alt="Mava ff mappo speed comparison" style="width:39.33%; display:inline-block; margin-right: 10px;"/>
    </a>
    <br>
    <div style="text-align:center; margin-top: 10px;"> Mava steps per second scaling with increased vectorised environments and total training run time for 20M environment steps.</div>
</div>


[Paper]: https://arxiv.org/pdf/2107.01460.pdf
[quickstart]: https://github.com/instadeepai/Mava/blob/develop/examples/Quickstart.ipynb
[jumanji]: https://github.com/instadeepai/jumanji
[cleanrl]: https://github.com/vwxyzjn/cleanrl
[purejaxrl]: https://github.com/luchris429/purejaxrl
[jumanji_rware]: https://instadeepai.github.io/jumanji/environments/robot_warehouse/
[jumanji_lbf]: https://github.com/sash-a/jumanji/tree/feat/lbf-truncate
[epymarl]: https://github.com/uoe-agents/epymarl
[anakin_paper]: https://arxiv.org/abs/2104.06272
[rware]: https://github.com/semitable/robotic-warehouse
[jaxmarl]: https://github.com/flairox/jaxmarl
[toward_standard_eval]: https://arxiv.org/pdf/2209.10485.pdf
[marl_eval]: https://github.com/instadeepai/marl-eval
[smax]: https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/smax
