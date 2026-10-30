# Fig. 8(a) 五种真实资源分配方法

| Method | Mean wCRB | TTC-AO relative gain (%) | Feasible cycles |
| --- | --- | --- | --- |
| TTC-AO | 0.00132103450549 | — | 401 |
| RU-AO | 0.00140403999191 | 5.911903 | 401 |
| DA-AO | 0.00144731206554 | 8.724971 | 401 |
| TTC-PD | 0.00135098935646 | 2.217253 | 401 |
| UD | 0.00166327515996 | 20.576310 | 401 |

详细实现、公式、相同预算和共同评价口径、外层精度、原实现差异及结论边界见[实验报告](experiment_report.md)。

运行：`python run_five_methods.py --sumo-home "C:\Program Files (x86)\Eclipse\Sumo"`。PyCharm先安装requirements.txt。完成目录受保护，重新计算请复制代码与source_snapshot到新目录。

推荐直接运行`rerun.py`：自动复制源代码及只用于事后对比的reference_baseline，在新时间戳目录重新执行SUMO和全部五方法。数学公式、变量与单位见[公式说明](formula_and_variables.md)。

五张方法曲线共同位于fig8a_five_methods.png/.pdf/.svg；401行共同指标见fig8a_five_methods.csv；各方法逐周期系统资源见method_cycles.csv；8020条目标级驻留、优化权重、共同TTC权重、CRB及FIM见resource_allocation_details.csv；统计见method_statistics.csv；可行性见constraint_checks.csv；原始SUMO轨迹见traffic.csv及warmup_traffic.csv；每个候选与AO更新见candidate_evaluations.csv和ao_trace.csv。

`per_method/`另含五个独立CSV，各自401行，合并该方法的系统资源、各目标驻留、未加权及共同TTC加权CRB、权重、求解器状态，便于单独检查。完整原始长表也保留。

本版是单次原始确定性SUMO场景的五种完整P2策略比较，没有30次扰动阴影，没有更换完整FIM或加入TTC精度约束。没有平滑、加偏移、删除周期或预设排序。图7、8(b)(c)、9及历史结果未覆盖。


当前正式图片统一保存于[fig8a_five_methods](../../图片/fig8a_five_methods.png)；源CSV仍在本实验目录。重复图片与冻结输入副本已按SHA-256去重，旧路径由[索引](../deduplication.json)解析。
