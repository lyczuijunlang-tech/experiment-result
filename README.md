# 车载 ISAC 递推资源分配与后验 TTC-CRB 约束

车载雷达利用上一周期的后验协方差预测目标状态，P1 在风险加权后验 CRB 目标下分配各目标驻留时间，P2 更新系统感知时间和功率。本版实验在 P1 中对每个预测为闭合的目标施加相对后验 TTC-CRB 上限。

实验使用 16 组配对噪声种子，每组 401 个正式周期。预设 5% 主门限下，6,416 个正式周期全部可行，平均风险加权测量 CRB 相对同周期基准增加 0.0666%。2% 严门限下有 879 个周期不可行，10% 宽门限下全部可行。这里的百分比门限是相对 TTC 标准差下界尺度。

- [完整实验报告、公式和复现步骤](TTC-CRB_constraint_ablation/README.md)
- [五张结果图，含 PNG、PDF、SVG](TTC-CRB_constraint_ablation/figures/)
- [门限敏感性汇总](TTC-CRB_constraint_ablation/threshold_summary.csv) · [逐目标对比](TTC-CRB_constraint_ablation/paired_target_comparison.csv) · [不可行周期](TTC-CRB_constraint_ablation/infeasible_cycles.csv)
- [逐周期原始数据](TTC-CRB_constraint_ablation/cycles.csv) · [逐目标原始数据](TTC-CRB_constraint_ablation/targets.csv) · [运行配置](TTC-CRB_constraint_ablation/config.json)
- [复现源码与 SUMO 输入](TTC-CRB_constraint_ablation/实验/)
