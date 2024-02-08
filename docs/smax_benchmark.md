# StarCraft Multi-Agent Challenge in JAX

We trained Mava’s recurrent systems on eight SMAX scenarios. The outcomes were then compared to the final win rates reported by [Rutherford et al., 2023](https://arxiv.org/pdf/2311.10090.pdf). To ensure fair comparisons we also train Mava's system up to 10 million timesteps with 64 vectorised environments.

</br>
</br>

<p align="center">
 <img src="images/smax_results/legend.png" alt="legend" width="50%"/>
</p>

<table width="100%">
<tr>
    <td width="33%"><img src="images/smax_results/2s3z.png" alt="2s3z" style="width: 100%; max-width: 100%;"></td>
    <td width="33%"><img src="images/smax_results/3s_vs_5z.png" alt="3s_vs_5z" style="width: 100%; max-width: 100%;"></td>
    <td width="33%"><img src="images/smax_results/3s5z_vs_3s6z.png" alt="3s5z_vs_3s6z" style="width: 100%; max-width: 100%;"></td>
</tr>
<tr>
    <td align="center"><code>2s3z</code></td>
    <td align="center"><code>3s_vs_5z</code></td>
    <td align="center"><code>3s5z_vs_3s6z</code></td>
</tr>
<tr>
    <td width="33%"><img src="images/smax_results/3s5z.png" alt="3s5z" style="width: 100%; max-width: 100%;"></td>
    <td width="33%"><img src="images/smax_results/5m_vs_6m.png" alt="5m_vs_6m" style="width: 100%; max-width: 100%;"></td>
    <td width="33%"><img src="images/smax_results/6h_vs_8z.png" alt="6h_vs_8z" style="width: 100%; max-width: 100%;"></td>
</tr>
<tr>
    <td align="center"><code>3s5z</code></td>
    <td align="center"><code>5m_vs_6m</code></td>
    <td align="center"><code>6h_vs_8z</code></td>
</tr>
<tr>
    <td width="33%"><img src="images/smax_results/10m_vs_11m.png" alt="10m_vs_11m" style="width: 100%; max-width: 100%;"></td>
    <td width="33%"><img src="images/smax_results/27m_vs_30m.png" alt="27m_vs_30m" style="width: 100%; max-width: 100%;"></td>
    <td width="33%"></td> <!-- Empty cell for alignment -->
</tr>
<tr>
    <td align="center"><code>10m_vs_11m</code></td>
    <td align="center"><code>27m_vs_30m</code></td>
    <td align="center"></td> <!-- Empty cell for alignment -->
</tr>
</table>
