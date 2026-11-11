# 车载 ISAC 实验图

各图提供 PNG、PDF、SVG。图8横坐标依次为 t(s)、d(m)、v(m/s)，纵轴为加权后验下界 ωPCRB。

| 图 | 源数据 |
| --- | --- |
| 图7，五场景优化收敛 | [fig7.csv](../结果/fig7.csv) |
| 图8(a)，五方法时域对比 | [fig8a.csv](../结果/fig8a.csv) |
| 图8(b)，距离扫描 | [fig8b.csv](../结果/fig8b.csv) |
| 图8(c)，径向闭合速度扫描 | [fig8c.csv](../结果/fig8c.csv) |
| 图9，逐目标贡献 | [fig9.csv](../结果/fig9.csv) |
| 图10(a)，历史反馈开关的总 ωCRB 对比 | [fig10a_total_wcrb.csv](../结果/fig10a_total_wcrb.csv) |

图10(a)比较 TTC-AO 调度是否使用历史后验协方差。两组均保留连续跟踪，以 16 组配对噪声回放的逐周期平均值绘图，包含 0–400 周期。启用反馈后，全程平均 ωCRB 降低 6.61%。
