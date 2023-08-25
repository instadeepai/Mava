# Jumanji RWARE Performance

There is a core difference in the way collisions are handled in the stateless JAX-based implementation of RWARE (called RobotWarehouse) found in [Jumanji][jumanji_rware] and the [original RWARE][original_rware] environment.

As mentioned in the original repo, collisions are handled as follows:
 > The dynamics of the environment are also of particular interest. Like a real, 3-dimensional warehouse, the robots can move beneath the shelves. Of course, when the robots are loaded, they must use the corridors, avoiding any standing shelves.
>
>Any collisions are resolved in a way that allows for maximum mobility. When two or more agents attempt to move to the same location, we prioritise the one that also blocks others. Otherwise, the selection is done arbitrarily. The visuals below demonstrate the resolution of various collisions.

In contrast to the collision resolution strategy above, the current version of the Jumanji implementation will not handle collisions dynamically but instead terminates an episode upon agent collision. In our experience, this appeared to make the task at hand more challenging and made it easier for agents to get trapped in local optima where episodes are never rolled out for the maximum length.

To investigate this, we ran our algorithms on a version of Jumanji's RWARE where episodes do not terminate upon agent collision, but rather multiple agents are allowed to occupy the same grid position. This setup is not identical to that of the original environment but represents a closer version to its dynamics, allowing agents to easily reach the end of an episode.

<p align="center">
    <a href="images/jumanji_vs_rware/ff_mappo_tiny2ag.png">
        <img src="images/jumanji_vs_rware/ff_mappo_tiny2ag.png" alt="Mava ff mappo tiny 2ag" width="30%" style="display:inline-block; margin-right: 10px;"/>
    </a>
    <a href="images/jumanji_vs_rware/ff_mappo_tiny4ag.png">
        <img src="images/jumanji_vs_rware/ff_mappo_tiny4ag.png" alt="Mava ff mappo tiny 4ag" width="30%" style="display:inline-block; margin-right: 10px;"/>
    </a>
    <a href="images/jumanji_vs_rware/ff_mappo_small4ag.png">
        <img src="images/jumanji_vs_rware/ff_mappo_small4ag.png" alt="Mava ff mappo small 4ag" width="30%" style="display:inline-block; margin-right: 10px;"/>
    </a>
    <br>
    <div style="text-align:center; margin-top: 10px;"> Mava MAPPO performance on Jumanji Rware with and without collision termination on tiny-2ag, tiny-4ag and small-4ag.</div>
</p>

<p align="center">
    <a href="images/jumanji_vs_rware/ff_ippo_tiny2ag.png">
        <img src="images/jumanji_vs_rware/ff_ippo_tiny2ag.png" alt="Mava ff mappo tiny 2ag" width="30%" style="display:inline-block; margin-right: 10px;"/>
    </a>
    <a href="images/jumanji_vs_rware/ff_ippo_tiny4ag.png">
        <img src="images/jumanji_vs_rware/ff_ippo_tiny4ag.png" alt="Mava ff mappo tiny 4ag" width="30%" style="display:inline-block; margin-right: 10px;"/>
    </a>
    <a href="images/jumanji_vs_rware/ff_ippo_small4ag.png">
        <img src="images/jumanji_vs_rware/ff_ippo_small4ag.png" alt="Mava ff mappo small 4ag" width="30%" style="display:inline-block; margin-right: 10px;"/>
    </a>
    <br>
    <div style="text-align:center; margin-top: 10px;"> Mava IPPO performance on Jumanji Rware with and without collision termination on tiny-2ag, tiny-4ag and small-4ag.</div>
</p>


[jumanji_rware]: https://instadeepai.github.io/jumanji/environments/robot_warehouse/
[original_rware]: https://github.com/semitable/robotic-warehouse
