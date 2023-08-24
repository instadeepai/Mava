# Jumanji RWARE Performance

There is a core difference in the way the logic of collisions is handled in the stateless JAX-based implementation of RWARE found in [Jumanji][jumanji_rware] and the [original RWARE repo][original_rware].

As mentioned in the original repo, collisions are handled as follows:
 > The dynamics of the environment are also of particular interest. Like a real, 3-dimensional warehouse, the robots can move beneath the shelves. Of course, when the robots are loaded, they must use the corridors, avoiding any standing shelves.
>
>Any collisions are resolved in a way that allows for maximum mobility. When two or more agents attempt to move to the same location, we prioritise the one that also blocks others. Otherwise, the selection is done arbitrarily. The visuals below demonstrate the resolution of various collisions.

The current version of the Jumanji implementation will not handle collisions dymanically but episodes will terminate upon agent collision. In our experience, this appeared to make the task at hand more challenging and made it easier for agetns to get trapped in local optima where episodes are never rolled out for the maximum length.

To investigate this, we also run our algorithms on a version of Jumanji's RWARE where the episode does not terminate upon agent collisions.


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
</p>




[jumanji_rware]: https://instadeepai.github.io/jumanji/environments/robot_warehouse/
[original_rware]: https://github.com/semitable/robotic-warehouse
