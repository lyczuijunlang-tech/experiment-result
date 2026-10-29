# 当前时域图导出依据

图8(b)(c)和图9直接由当前五方法实际SUMO运行的TTC-AO目标记录和车辆状态导出。距离、径向闭合速度及逐目标wCRB全部使用原始逐周期值，未平滑或插值。

每目标401个时间点，k1至k4编号与五方法资源明细一致。图9四条贡献之和等于TTC-AO系统总wCRB。历史独立重运行轨迹与当前五方法轨迹逐点一致，详见当前实验包baseline_comparison.json及independent_formula_checks.json。
