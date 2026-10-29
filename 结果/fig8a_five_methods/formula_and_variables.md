# 公式、变量及代码映射

## 当前交通状态与物理风险

单个ego、四个固定目标。每个0.1 s资源决策周期读取同一份真实SUMO当前状态。车辆中心位置由SUMO前保险杠坐标减去半车长方向向量取得；速度使用原后向位置差分定义。预热状态也已保存，首周期可以独立重构差分。

`r_k=p_k−p_ego`，`d_k=||r_k||₂`，`v_k=−(V_k−V_ego)ᵀr_k/d_k`。正v表示闭合；φ是视线在ego朝向坐标系中的方位。

`TTC_k=d_k/v_k`（v>0），否则TTC=∞。原风险原值`raw_k=1/(TTC_k+ε_TTC)`，`ε_TTC=0.1 s`；`ωTTC_k=K raw_k/Σraw`，K=4。若原值全零，回退到四个权重均为1。风险权重在当前资源决策周期内固定，不使用未来状态或已优化资源生成风险权重。

代码：`source_snapshot/main.py::vehicle_states_from_traci`、`source_snapshot/model.py::relative_observation`、`compute_ttc_weights`。

## 原始雷达、EFIM与无量纲CRB

匹配波束：`w_k=a_t(φ_k)/||a_t||`，`b_rx,k=a_r(φ_k)/||a_r||`；每周期重算。Tx/Rx各4阵元，半波长间距，各理想匹配增益为4。

单基地雷达：

`Pr,k = Ps Gt,k Gr,k λ² σ / [(4π)³ d_k⁴ L]`。

fc=77 GHz，λ=c/fc，c=299792458 m/s；雷达带宽B=200 MHz，σ=10 m²，L=2。噪声谱密度`N0=kB Tnoise F`，kB=1.380649×10⁻²³ J/K，Tnoise=290 K，F=10^(6/10)。原LFM脉冲时长5 μs。

保持原未知复幅度消元和中心化快频率/慢时间统计模型，对角EFIM为：

`Jdd=8π²(Pr/N0)(2/c)²(B²/12)Tk`，

`Jvv=8π²(Pr/N0)(2fc/c)²Tk³/12`，`Jdv=0`。

`CRBd=1/Jdd`，单位m²；`CRBv=1/Jvv`，单位(m/s)²。无量纲标量`Ck=CRBd/d_ref²+CRBv/v_ref²`，参考方差分别为1 m²与1 (m/s)²。

代码：`closed_form_beams`、`radar_received_power_w`、`theoretical_lfm_echo`、`fim_and_crb`、`scalar_crb_coefficients`。原物理文件没有修改，没有更换完整FIM，也没有加入TTC精度约束。

## 五种目标级规则及系统目标

| 方法 | 分配规则/权重 | P2优化目标 |
| --- | --- | --- |
| TTC-AO | 完整原P1；ωTTC | ΣωTTC Ck |
| RU-AO | 完整原P1；所有优化权重1 | ΣCk |
| DA-AO | 完整原P1；raw=χ/(d+εd)，归一化到和K；全零回退为1 | ΣωDA Ck |
| TTC-PD | Tk=Tmin+(Ts−KTmin)ωTTC_k/ΣωTTC | ΣωTTC Ck |
| UD | Tk=Ts/K，不使用风险作为目标级时间优先级 | ΣωTTC Ck |

χ=1当v>0，否则0，与TTC原风险资格条件一致。εd=0.1 m在运行前固定；DA资格判定之外不使用闭合速度大小。UD系统目标仍是共同风险指标，但其目标级驻留为均匀，不能把资源明细中的系统目标权重解读为UD的驻留优先级。

完整原P1最小化给定权重下的ΣωCk，约束ΣTk=Ts、Tk≥Tmin。原系数形式`Ck=a_k/(Ps Tk)+b_k/(Ps Tk³)`，用100次KKT对偶二分完整求解，每个候选重新求解，不使用旧驻留缓存。

RU原实现常数权重1/K只改变目标尺度，不改变理论最优解；本次RU使用1，保证与TTC/DA的权重总和K尺度相同。5个固定审查周期额外比较1与1/K的完整AO结果，结果记录于`RU_scale_checks.csv`，不用于改变资源决策。

新增代码：`allocation_methods.py::optimization_weights`、`direct_dwell`、`allocate_at`。PD/UD直接计算，不声称调用了最优P1。其KKT字段不适用，输出为空；兼容原AO接口的inner evaluation计数与真实完整P1求解次数明确区分。

## 相同系统可行域与AO设置

系统帧T=50 ms；每目标最低驻留Tmin=2 ms。因此8 ms≤Ts≤T−Tc,min。

通信最低时长`Tc,min=T Rmin/[Bc log2(1+γc)]`，Bc=20 MHz，Rmin=10 Mbps。γc按原宏小区+RSU辅助空间信道从当前ego位置计算，逐周期记录；不增加随机阴影扰动。

`0≤Ps≤Ps,max=50 mW`，`Ps Ts≤Es,max=1 mJ`。本次5方法系统资源选择范围完全相同。

每种方法独立执行：固定Ps优化Ts→固定新Ts优化Ps；内层/分配规则每个候选重新调用。使用原AO同一代码对象，保留原端点/当前点检查、松弛规则和停止判断：首次Ts=25 ms、Ps=13.5 dBm；时间松弛0.35，功率松弛0.03，接近时间边界时0.15；η=10⁻⁶，最多100轮；可行性容差10⁻¹⁰；时域热启动仅用各自上一周期Ts/Ps和原0.995功率回退。

原scalar_max_iterations=24、time_search_tolerance=10⁻⁷s、power_search_tolerance=10⁻⁸W也保留。当前块求解由模型单调性下的端点/当前点评价实现，不能把保留的配置项误称为实际执行了通用标量迭代24轮。

## 统一最终指标与统计

五种方法最终只能由`allocation_methods.py::evaluate_common_ttc_wcrb`生成比较指标：

`J_m(t)=ΣωTTC_k(t) Ck(Tk^(m)(t), Ps^(m)(t))`。

这里的ωTTC在同一周期五方法完全相同。RU/DA自己的优化权重只参与资源决策，不能替代最终物理TTC权重。`allocation_objective`列是各自决策目标，不用作不同方法间wCRB比较。

共401个实际时间点t=0,0.1,…,40 s，全时域均值`Jbar_m=Σ_n J_m(t_n)/401`。TTC-AO相对基线m改善为`100(Jbar_m−Jbar_TTC)/Jbar_m`。这不是逐周期百分比的均值。

原功率零端点Ps=0对应零FIM及+∞目标，是被明确标记的不可辨识候选；最终解必须正功率、FIM正定、CRB有限。没有将退化候选改成有限小值。

## 文件中的主要单位

`Ts_s`、`dwell_s`：s；`Ps_w`：W；`energy_j`：J；`communication_rate_bps`：bit/s；`distance_m`：m；`closing_speed_mps`：m/s；`CRB_d_m2`：m²；`CRB_v_m2ps2`：(m/s)²；`scalar_crb`和`final_ttc_weighted_crb`：无量纲；`total_wcrb`：四目标共同风险贡献之和。

`reference_baseline/`只供新仿真完成后的差异核对，任何方法求解都不读取其中的CSV。所有主图数据来自本目录新生成的`traffic.csv`、`method_cycles.csv`及目标级资源明细。
