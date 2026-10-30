# 正式图片与源数据

每张仅保留一套PNG、PDF、SVG；不同格式用于预览与论文排版，不作为重复文件删除。图8三联图为独立排版，也保留。原始CSV只在下表路径存储一次。

| 图片文件名（扩展名PNG/PDF/SVG） | 唯一源数据 |
| --- | --- |
| fig7_multi_scenario_ao_convergence | [AO记录](../结果/fig7_latest/fig7_convergence.csv) |
| fig8a_five_methods | [时间扫描](../结果/fig8a_five_methods/fig8a_five_methods.csv) |
| fig8b_five_methods | [距离扫描](../结果/fig8bc_five_methods/fig8b_five_methods.csv) |
| fig8c_five_methods | [闭合速度扫描](../结果/fig8bc_five_methods/fig8c_five_methods.csv) |
| fig8_abc_combined | 上述图8(a)(b)(c)三份源数据 |
| fig9_ttc_aware_target_wcrb | [逐目标贡献](../结果/trajectory_figures/fig9_ttc_aware_target_wcrb.csv) |

图8横轴依次为`t(s)`、`d(m)`、`v(m/s)`，纵轴统一`ωCRB`。均保留原始样本，无插值或平滑。本次只改标签，不重新宣称执行SUMO实验。

[已删除的重复文件及保留位置](../结果/deduplication.json)。旧文件版本可从Git历史追溯。
