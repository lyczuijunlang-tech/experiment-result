# 车载 ISAC 后验 TTC-CRB 约束消融

本实验在原递推 Proposed TTC-aware 的 P1 中加入逐目标后验 TTC 精度约束。E1 是重新运行的原方法，E2 使用预设 5% 主门限，E3 使用 2% 和 10% 两档。代码沿用原状态预测、TTC 风险权重、LFM 雷达 FIM、后验协方差递推、风险加权后验目标和 P2 资源约束。正式图7–10(a)及其数据仍在原仓库版本，本目录只记录独立消融。

## 模型与约束

在线调度取预测距离 \(\hat d^-\) 和预测径向闭合速度 \(\hat v^-\)。对 \(\hat v^->0\)，\(\widehat{\mathrm{TTC}}=\hat d^-/\hat v^-\)，其梯度为 \(g=[1/\hat v^-, -\hat d^-/(\hat v^-)^2]^\top\)。利用原有后验协方差 \(P_k^+=[(P_k^-)^{-1}+J_k]^{-1}\)，定义

$$
\operatorname{PCRB}_{\mathrm{TTC},k}=g_k^\top P_k^+g_k,\qquad
\Gamma_k^+=\frac{\operatorname{PCRB}_{\mathrm{TTC},k}}{\widehat{\mathrm{TTC}}_k^2}
=\frac{P_{dd,k}^+}{(\hat d_k^-)^2}
-\frac{2P_{dv,k}^+}{\hat d_k^-\hat v_k^-}
+\frac{P_{vv,k}^+}{(\hat v_k^-)^2}.
$$

P1 继续最小化 \(\sum_k\omega_k\operatorname{tr}(WP_k^+)\)，同时满足 \(\sum_k T_{s,k}=T_s\)、\(T_{s,k}\ge T_{\min}\) 及每个闭合目标的 \(\Gamma_k^+\le\bar\Gamma\)。闭合速度为零或负数时，TTC 约束不适用，对应字段留空。SUMO 目标真值用于模拟观测和离线评分，在线 TTC 约束使用预测状态。

计算采用 \(u=[\hat v^-/\hat d^-,-1]^\top\)，直接判定 \(u^\top P^+u\le\bar\Gamma(\hat v^-)^2\)。该形式与上式等价，避免在求解约束时使用速度倒数。原始速度全程保留。\(0<\hat v^-\le0.05\,\mathrm{m/s}\) 只标记为接近零的诊断事件，仍按同一公式判定。

固定系统时间和功率后，增加该目标驻留时间会增大 \(J_k\)，所以 \(\Gamma_k^+\) 非增。代码用求根计算每个目标的最低所需驻留时间 \(T_{s,k}^{\rm req}\)，将它作为 P1 的硬下限。若一个目标用尽可单独获得的时间仍无法达标，或所有目标的下限之和超过 \(T_s\)，候选不可行。P2 先检查最大允许感知时间及其可用功率。原 FIM 下，在预算内把驻留时间同比放大到时间上限并相应调整功率，可以保持或提高两项测量信息。因此该点不可行时，整个周期无可行资源分配。可行时沿用原来的时间和功率更新顺序，跳过不可行的中间候选。

不可行周期写入失败表，按预测状态和先验协方差继续递推，感知资源与单周期 \(\omega\mathrm{CRB}\) 留空。该周期未获得新的感知测量或方位读数。包含所有周期的后验口径另列一列，失败周期使用纯预测协方差。数学可行性按同一固定门限判定，最终矩阵复核使用相对数值容差 \(10^{-9}\)。

## 配置与结果

使用原 40 s SUMO 记录，决策间隔 0.1 s，四个固定目标，RCS 均为 10 m²，目标最低驻留 2 ms。16 个配对噪声种子为 17001–17016，每组包括 19 次预热更新和 401 个正式周期。四个实验组各有 6,416 个正式周期。传感器、过程噪声、功率和能量等具体取值见 [config.json](config.json)。

门限在运行前按相对标准差下界尺度设定。5% 为主实验，2% 为更严格的压力测试，10% 为较宽松的参照。同一组所有目标共用相应的 \(\bar\Gamma\)。表中的百分数描述理论方差下界的平方根尺度。

| 实验组 | \(\bar\Gamma\) | 正式可行周期 | 平均 \(\omega\mathrm{CRB}\) | 平均 \(\omega\mathrm{PCRB}\) | 相同周期相对 E1 的变化 |
| --- | ---: | ---: | ---: | ---: | --- |
| E1 原 Proposed TTC-aware | — | 6,416/6,416 | 0.0013789744 | 0.0001571990 | 参照 |
| E2 5% 主门限 | 0.0025 | 6,416/6,416 | 0.0013798928 | 0.0001572459 | +0.0666% / +0.0298% |
| E3 2% 严门限 | 0.0004 | 5,537/6,416（86.30%） | 0.0014708716 | 0.0001728590 | +7.4446% / +10.5033% |
| E3 10% 宽门限 | 0.0100 | 6,416/6,416 | 0.0013789742 | 0.0001572001 | −0.000018% / +0.000703% |

上表只对各组的可行正式周期计算指标，变化列使用相同种子和周期的 E1 值。\(\omega\mathrm{CRB}\) 来自实际模拟测量 FIM，\(\omega\mathrm{PCRB}\) 来自跟踪后验矩阵。两项离线评分均沿用共同的物理 TTC 权重。所有成功分配的周期均满足通信 QoS 和能量约束。

5% 门限下，原方案有 535 个周期的 k3 超出该门限。加入约束后全部正式周期达标，k3 有 538 个目标周期达到约束边界。k3 的最大 \(\Gamma^+\) 从 0.00421314 降至 0.0025，平均驻留从 8.3425 ms 增至 8.3675 ms。其余目标平均驻留分别变化 −0.0071、−0.0113、−0.0066 ms，系统平均感知时间和功率分别为 43.00945 ms 和 23.25074 mW，与 E1 相同。10% 门限在正式周期中未成为活跃约束。

2% 门限共有 879 个正式周期失败。其中 268 次为 k4 在可单独取得的最大驻留时间下仍无法达标，611 次为各目标最低所需驻留时间之和超过预算。另有 57 次预热失败。种子 17008 和 17012 的 401 个正式周期均未能获得可行分配，种子 17011 在 7.7 s 首次恢复。图中的 7.7 s 跳变对应可行种子数从 13 增至 14。计入纯预测失败周期后，2% 组全时域平均 \(\omega\mathrm{PCRB}\) 为 68.4249，反映无测量更新时的协方差积累。该值与上表的成功分配条件均值口径不同。

该消融检验目标级理论下界和资源可行性。实际 TTC 估计误差及 RMSE 需要独立实验。

## 文件与复现

| 文件 | 内容 |
| --- | --- |
| [cycles.csv](cycles.csv) | 26,880 条周期记录，含预热、P2 候选拒绝、系统预算裕量、wCRB 与 wPCRB |
| [targets.csv](targets.csv) | 107,520 条目标记录，含预测状态、后验矩阵、TTC-CRB、相对下界、裕量和驻留时间 |
| [threshold_summary.csv](threshold_summary.csv) | 门限敏感性、可行比例、同周期 E1 对比、QoS 和能量满足率 |
| [target_summary.csv](target_summary.csv) 和 [paired_target_comparison.csv](paired_target_comparison.csv) | 各目标驻留与 TTC-CRB 的配对对比 |
| [infeasible_cycles.csv](infeasible_cycles.csv) | 含预热的全部失败周期、失败原因和 P2 上限 |
| [verification.json](verification.json) | E1 与发布数据核对、边界目标计数、近零速度和各组可行性复核 |
| [figures](figures/) | 逐时间 wCRB、wPCRB、可行比例、逐目标相对下界和驻留时间，均提供 PNG、PDF、SVG |

图中每个时间点对该组成功分配的种子取平均。可行性图则以 16 组种子为固定分母，目标下界图只包含当期闭合且成功分配的目标。5% 与 10% 部分曲线重合是数据结果。CSV 的 `cycle` 从第一条预热记录开始，正式周期编号为 `cycle - 19`。

修改文件位于 `实验/latest/recursive_feedback/`。修改 `run_feedback.py`，把最低所需驻留时间接入原 P1/P2 并显式记录不可行周期。新增 `ttc_precision.py`（约束和可行性）、`test_ttc_precision.py`（五项单周期验证）、`run_ttc_ablation.py`（四组配对回放和汇总）及 `plot_ttc_ablation.py`（从新 CSV 画图）。其余随包代码和 SUMO 输入是复现依赖。

解压后在根目录执行：

```powershell
python -m pip install -r 实验/requirements.txt
python -m unittest discover -s 实验/latest/recursive_feedback -p test_ttc_precision.py -v
python 实验/latest/recursive_feedback/run_ttc_ablation.py --output rerun --workers 8
python 实验/latest/recursive_feedback/plot_ttc_ablation.py rerun
```

原始无约束回放与已发布逐周期数据的最大绝对差为 \(8.68\times10^{-19}\)（\(\omega\mathrm{CRB}\)）和 \(5.43\times10^{-20}\)（\(\omega\mathrm{PCRB}\)）。五项单周期测试、五张图的 PDF 字形与碰撞检查均已通过。
