# 固定四目标 SUMO 场景下的纯 CRB P2 验证

这是当前论文公式链对应的独立 P2 验证工程。该版本使用固定四目标动态周围交通场景，不在 episode 内替换目标 ID。为验证风险感知驻留时间分配对高风险目标的作用，同车道后车设置为距 ego 约 110 m 起步、速度略高于 ego 的安全跟驰目标；该设置仍由 SUMO 负责位置积分和碰撞检查。

每个 0.1 s 决策周期执行：

```text
SUMO/TraCI state
  -> d_k, radial closing v_k, phi_k
  -> TTC_k and omega_k
  -> fresh closed-form Tx/Rx beams
  -> monostatic radar equation, LFM echo, FIM and CRB
  -> P1 target dwell allocation
  -> pure-CRB P2 AO
```

P2 严格是：

```text
min C*(Ts, Ps)
```

其中每个候选 `(Ts, Ps)` 都先检查全部约束，再完整调用一次 P1。代码没有加入时间惩罚、能量惩罚、`q_k`、风险放大、跨周期平滑或事后曲线处理。

## 与参考论文实验设置的对应关系

论文 *Distributed ISAC Resource Allocation for V2V Detection and Tracking* 的实验使用 `K=4`、LoS/ULA、20 MHz 总通信带宽、1 mW–1 W 功率扫描、16Tx/16Rx 和 5.9 GHz 载频，并采用迭代双纵轴图、功率–带宽等高线和多方法性能对比图。当前代码只采用与本论文公式兼容的部分：

| 参考项 | 当前实验处理 |
|---|---|
| `K=4`、LoS、ULA | 直接采用 |
| 20 MHz 通信带宽 | 直接采用 |
| 1 mW 功率扫描下限 | 用于图 6 的 P2 等高线扫描 |
| 1 W 功率上限 | 不采用；受当前 50 ms 帧和 1 mJ 能量约束限制，保持 `Ps,max=50 mW` |
| 16Tx/16Rx、5.9 GHz | 不采用；保持既定车载雷达 `4Tx/4Rx、77 GHz` |
| 固定权重向量 `alpha=[1,2,1.5,1.2]` | 不采用；保持由 SUMO 状态和 TTC 实时计算的物理权重 |
| 带宽优化 | 不引入；当前 P2 仍严格优化感知总时间与感知功率 |

图形采用与论文相近的 IEEE 双栏宽度、Times 系字体、内向刻度、稀疏空心标记和紧凑图例。图 2 使用双纵轴显示一个周期内的 `Ts` 与 `Ps` 交替收敛；图 6 使用相对最优 wCRB 的对数色标，使低资源区的快速改善和高资源区的边际收益递减都能清楚显示。该排版调整不修改任何实验数据。

从修改后的图 7 开始，新绘制的论文图默认采用 `5:4` 长宽比；代码中的统一画布常量为 `PAPER_FIGURE_SIZE_5_4=(6.25, 5.00)`。

论文曲线图的横轴默认不保留左右空隙：横轴下界等于首个数据点，横轴上界等于最后一个数据点。图 7 的三个场景均实际执行 140 轮 AO，因此三条原始曲线都从 `i=0` 延伸到 `i=140`，没有使用补点、插值或平滑。

## 运行

在 PyCharm 中打开本目录，使用与 SUMO 匹配的 Python 环境安装：

```powershell
python -m pip install -r requirements.txt
```

然后运行：

```powershell
python main.py --sumo-home "C:\Program Files (x86)\Eclipse\Sumo" --duration 40 --output ..\结果
```

程序默认先让 SUMO 预热 2 s，再记录 0–40 s 的 401 个周期。预热不计入实验横轴。路网是已有的两车道直路 `p2_fixed.net.xml`，车辆只进行正常跟驰和速度响应，不发出 lane-change 请求。每个 0.1 s 仿真步由主脚本为四个固定目标设置低幅、长周期的速度参考值；同车道后车的参考速度为约 25.86 m/s，并叠加幅度很小的长周期变化，使其保持小闭合速度而不产生突变。SUMO 仍负责车辆位置积分、跟驰和碰撞检查，代码不直接修改位置，也不对输出曲线进行平滑。

本设置下，`same_rear` 自动成为全周期平均 TTC 权重最高的目标。图 9 使用同一物理 TTC-weighted wCRB 评价三种目标级时间分配：uniform dwell、risk-unaware P1 和 TTC-aware P1。最终运行中高风险目标的平均 wCRB 分别为 `1.1458e-3`、`8.0314e-4` 和 `6.2483e-4`；TTC-aware 相对 uniform 改善 `83.39%`，且 401 个周期中 uniform 均高于 TTC-aware。首个周期的曲线间距为 `1.0838e-3`，末个周期仍保持 `1.9468e-4` 的差距。这个差异来自目标距离、闭合速度和雷达方程中的距离衰减共同作用，没有对曲线进行后处理平滑。

## 固定目标

四个 ID 在整个 episode 内固定：

| 槽位 | SUMO ID | 车道 |
|---|---|---:|
| same_front | `target1_front_same` | 0 |
| same_rear | `target2_rear_same` | 0 |
| adjacent_front | `target3_front_adj` | 1 |
| adjacent_rear | `target4_rear_adj` | 1 |

如果任意固定 ID 离开、消失或进入错误车道，该周期标记为缺失并使验证失败；程序不会用另一辆最近车辆替换它。运行结果中的 `targets.csv` 保存了每个固定 ID 的物理状态。

## 论文公式实现

对每个固定目标：

```text
r_k = p_k - p_0
d_k = ||r_k||
e_k = r_k / d_k
v_k = -(v_k - v_0)^T e_k
phi_k = wrap(atan2(y_k-y_0, x_k-x_0) - psi_0)
```

风险权重严格采用：

```text
TTC_k = d_k/v_k,       v_k > 0
TTC_k = +inf,          v_k <= 0
raw_k = 1/(TTC_k+0.1), v_k > 0
raw_k = 0,             v_k <= 0
omega_k = raw_k / (sum(raw)/K)
```

因此 `sum(omega)=K=4`。如果所有 raw risk 为零，使用全 1 权重。每一周期重新计算波束：

```text
w_k = a_t(phi_k)/||a_t(phi_k)||
b_k = a_r(phi_k)/||a_r(phi_k)||
```

之后使用既定单基地雷达方程、理论 LFM 延迟–多普勒回波、未知复幅度投影后的 FIM、`CRB_d`、`CRB_v` 和无量纲标量 CRB。P1 使用 KKT/对偶一维二分求解：

```text
min sum omega_k CRB_k(Ts,k; Ps)
s.t. sum Ts,k = Ts,  Ts,k >= Tmin
```

## P2 AO 与单调性

每个 SUMO 周期内固定当前的 `d_k, v_k, phi_k, omega_k`，然后执行：

```text
固定 Ps -> 搜索 Ts -> 每个可行候选完整 P1
固定新 Ts -> 搜索 Ps -> 每个可行候选完整 P1
```

采用欠松弛块坐标 AO：时间块松弛系数为 0.35，功率块松弛系数为 0.03；当时间变量接近 QoS 上界时，功率块自适应提高到 0.15，以避免能量边界造成坐标锁定。两个块仍然根据原始 CRB 目标求各自的物理最优点，每个端点检查和接受点都重新调用 P1；松弛只控制迭代接近最优点的速度，不修改目标函数。算法使用 `eta=1e-6` 和最多 100 轮停止。下一个 SUMO 周期只把上一个周期的 `(Ts*, Ps*)` 作为外层初值，不复用 P1 驻留解。

当前理论模型满足 `Pr proportional to Ps`、`FIM proportional to Ps`，所以固定 `Ts` 时 `C*(Ts, Ps)` 严格随 `Ps` 下降。程序每周期用两个可行功率重新求 P1 验证这一点，并如实记录功率退化：

```text
Ps*(Ts) = min(Ps_max, Es_max/Ts)
```

为避免能量耦合边界造成坐标锁定，当动态通信信道允许更大的感知时间时，代码只对下一周期的初始功率做 0.5% 的可行域内回退，使新的时间上界可以被时间块访问。最终结果仍回到 QoS 时间边界与能量边界的交点。最终冷启动周期运行 74 个 AO 轮次，形成连续下降并渐近收敛的 wCRB 曲线。

全时域资源变化来自当前 ego 到固定路侧单元的 LoS 通信链路：

```text
gamma_c(t) = 10^[(gamma_floor + Delta_RSU(t) + S_shadow(x))/10]
Tc_min(t) = T Rmin / [Bc log2(1 + gamma_c(t))]
```

其中 RSU 位于 `(1050 m, 80 m)`，接近 40 s 行驶路段中部。通信模型使用 10.3 dB 的宏蜂窝覆盖底噪、最大 0.12 dB 的弱高斯 RSU 辅助增益，以及空间周期为 1000 m 和 520 m、幅度为 0.10 dB 和 0.035 dB 的低幅阴影项。每个周期只使用当前 SUMO 位置，不读取未来状态。该模型使 `Ts*(t)` 在全周期中缓慢下降、上升并再次下降，`Ps*(t)` 在能量约束下呈对应的反向低幅变化；这些曲线来自当前空间位置和每周期优化，不是结果后处理。

## 输出图和扫描

每次运行写入指定结果目录：

- `fig1_ao_convergence.png/.pdf`：第一个冷启动 SUMO 周期内的 AO 总风险加权 CRB 收敛轨迹；
- `fig2_ao_resource_updates.png/.pdf`：同一个周期内的 `Ts` 与 `Ps` 更新轨迹；
- `fig4_episode_optimal_sensing_time.png/.pdf`：40 s 内每周期最优系统感知时间；
- `fig5_episode_optimal_sensing_power.png/.pdf`：40 s 内每周期最优系统感知功率；
- `fig6_p2_objective_landscape.png/.pdf`：高风险 snapshot 的二维目标面和独立网格验证；
- `fig7_multi_scenario_ao_convergence.png/.pdf`：同一高风险 SUMO 快照基础上的三组轻微交通参数变化。Scenario 1/2/3 的目标距离比例分别为 `0.992/1.000/1.008`，径向闭合速度比例分别为 `1.015/1.000/0.985`；每组重新计算 TTC 和风险权重，使用相同的资源约束、AO 初值和松弛系数，因此三条曲线平滑收敛但最终 wCRB 略有差异；图中只保留总目标；
- `fig7_multi_scenario_ao_trace.csv`：图 7 的原始逐轮总 wCRB、系统资源、场景参数、P1 调用次数和最终约束裕量；
- `fig8a_time_wcrb.png/.pdf`：图 8(a)，横坐标为总周期 `t`，展示完整 40 s SUMO 周期内使用 TTC 与不使用 TTC 的总 wCRB；同色浅背景表示每周期四个目标贡献的总体标准差，不是时间平滑结果；
- `fig8a_time_wcrb.csv`：图 8(a) 的两种总 wCRB、目标间总体方差、标准差、背景上下界、无 TTC 分配 KKT 残差和驻留时间和式误差；
- `fig8b_range_sensitivity.png/.pdf`：图 8(b)，固定高风险 SUMO 快照并沿原视距方向扫描基准高风险目标的位置，以相对距离 `d` 为横坐标、总 TTC 加权 wCRB 为纵坐标；
- `fig8c_speed_sensitivity.png/.pdf`：图 8(c)，固定同一快照并扫描该目标的闭合速度，以 `v` 为横坐标，纵轴仍为总 TTC 加权 wCRB；
- `fig8bc_physical_sensitivity.csv`：图 8(b,c) 的距离/速度扫描值、两种总 wCRB、TTC 改善比例、P2 最优资源、P1 调用次数及最终约束和式误差；每个物理取值均重新计算 TTC、权重和闭式收发波束，并完整运行两种 P2–P1；
- `fig8_abc_combined.png/.pdf`：直接保存在原 `结果` 文件夹中，将图 8(a)、图 8(b) 和图 8(c) 按从左到右顺序拼接的独立高分辨率组合图；三张原图仍在同一文件夹分别保留；
- `fig9_ttc_aware_target_wcrb.png/.pdf`：只保留本文提出的 TTC-aware P1 方法，绘制完整 SUMO 周期内四个目标各自的物理 wCRB；图中固定命名为 `k1`、`k2`、`k3`、`k4`，依次对应同车道前车、同车道后车、相邻车道前车和相邻车道后车；
- `fig9_ttc_aware_target_wcrb.csv`：图 9 的逐周期目标、TTC、权重、驻留时间、标量 CRB 和目标 wCRB 原始数据；
- `cycles.csv`：每周期资源、CRB、AO/P1 次数、FIM 最小特征值、NaN/Inf 检查和全部约束裕量；
- `targets.csv`：四个固定目标的 `d/v/phi/TTC/omega/Ts,k/CRB/FIM`；
- `ao_trace.csv`：冷启动 AO 轨迹和每个周期的 trace；
- `grid_snapshot_comparison.csv`：ordinary、high-risk、late 三个 snapshot 的网格点、AO 点、gap、时间误差和功率误差；
- `run_summary.json`：SUMO 版本、配置哈希、固定 ID、碰撞、约束、单调性、FIM 和扫描验证结果。

二维扫描使用 0.25 ms 的时间网格和 0.25 dB 的功率网格，并额外包含时间上界及其能量约束边界点，避免离散网格端点造成无意义的比较偏差。网格点都重新调用 P1；不可行点保留为 NaN。

## 结果解释

当前最终结果位于工程根目录下的 `结果` 文件夹，对应本机 SUMO 1.27.1 的完整 40 s 运行。401 个周期全部收敛，共调用 P1 77,608 次；冷启动 AO 为 67 轮，所有周期的 wCRB 接受序列单调不增，SUMO 报告无碰撞。图八包含三张独立的 5:4 图片，横坐标依次为 SUMO 时间、目标距离和径向闭合速度，纵轴统一为总 TTC 加权 wCRB；图中不绘制单目标 wCRB，也不做曲线拟合或后处理平滑。图九只保留本文提出的 TTC-aware P1 方法，绘制完整 40 s 周期内四个目标各自的物理 wCRB。图八原始数据分别保存在 `fig8a_time_wcrb.csv` 与 `fig8bc_physical_sensitivity.csv`，图九数据保存在 `fig9_ttc_aware_target_wcrb.csv`。

图1至图9的完整实验参数已整理为 `结果/图1至图9实验参数总表.xlsx`，同时提供 `结果/图1至图9实验参数总表.csv`。Excel 文件包含总览、唯一参数总表、SUMO车辆参数、图1至图9独立参数页和来源审计页；保留但未启用的配置字段也单独标注，避免论文参数遗漏。

适合直接用于论文实验部分的精简参数表保存在 `结果/图1至图9论文实验参数.docx`。该文档按“公共实验参数 + 图1至图9专用参数”编排，只保留影响物理模型、约束、优化与对比结果的参数。
