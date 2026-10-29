# 图8(b)(c)：五方法物理敏感性实验

图8(a)保留原始完整时域五方法结果；本次只增加(b)距离—总wCRB、(c)径向闭合速度—总wCRB五方法对照，并同步三联图。

基线来自已实际执行SUMO的t=40.0s快照，高风险目标k2。距离基线71.2061986m，闭合速度基线0.977267956m/s。选点规则沿用原离线最大风险差快照，不用于在线决策读取未来。

距离取基线75%–125%，闭合速度取50%–150%，各51点，均包含基线。只控制被选目标，另三目标与ego冻结。距离修改真实状态坐标并重新投影；速度沿视线方向改变，保留切向分量。每点重新计算d、径向v、φ、TTC、风险权重和匹配波束，分别冷启动求解五种完整P2。

这些是受控静态反事实状态，不是重新执行102次SUMO，也不是将原轨迹按距离或速度排序后连线。速度扫描低于道路30m/s限速；反事实速度并非重新在原车型maxSpeed配置下积分出的速度。原(a)交通轨迹不变。

五方法与(a)完全一致：TTC-AO/RU-AO/DA-AO分别用TTC/等权/距离权重完整P1；TTC-PD/UD直接分配目标驻留，均保留完整外层AO。最终统一调用evaluate_common_ttc_wcrb，用该扫描点同一物理TTC权重与原对角EFIM/CRB评价。εTTC=0.1s，εDA=0.1m；初值Ts=25ms、Ps=13.5dBm，η=1e-6、最多100轮、原松弛0.35/0.03(近边界0.15)、最低驻留2ms、帧50ms、功率50mW、能量1mJ及通信QoS保持一致。

## 实算统计

| 扫描 | 方法 | 网格均值wCRB | TTC-AO相对改善 % | 可行/收敛点 |
| --- | --- | --- | --- | --- |
| range | TTC-AO | 0.000684413776692 | — | 51/51 |
| range | RU-AO | 0.000750566072684 | 8.813654 | 51/51 |
| range | DA-AO | 0.000761648972918 | 10.140524 | 51/51 |
| range | TTC-PD | 0.000732245251586 | 6.532166 | 51/51 |
| range | UD | 0.00082211287832 | 16.749415 | 51/51 |
| closing_speed | TTC-AO | 0.000665741892708 | — | 51/51 |
| closing_speed | RU-AO | 0.000732811315656 | 9.152345 | 51/51 |
| closing_speed | DA-AO | 0.000741006088312 | 10.157028 | 51/51 |
| closing_speed | TTC-PD | 0.000699499941654 | 4.826026 | 51/51 |
| closing_speed | UD | 0.000774970284438 | 14.094526 | 51/51 |

这里的平均值是均匀扫描网格上的描述统计，不是全时域平均，也不是交通条件的概率期望。所有真实交叉保留，见curve_crossings.csv，未根据排名调整参数。

速度图同时改变归一化物理TTC权重和优化后的驻留时间。因此，TTC-AO的总wCRB不必随闭合速度单调增加；下降不能直接解释为高速度本身让估计更容易，或更高风险代表更安全。原冻结几何EFIM本身不以速度大小改变信息系数，本图还包含风险重分配和资源优化的联合作用。

## 数值核查

510次新AO、2040条目标级分配，全部有效且早停收敛。独立几何、TTC、PD/UD和EFIM公式误差：{'distance_m': 0.0, 'radial_velocity_mps': 0.0, 'TTC_weights': 4.440892098500626e-16, 'optimization_weights': 4.440892098500626e-16, 'control_variable': 1.1368683772161603e-13, 'frozen_other_states': 0.0, 'scalar_crb': 2.168404344971009e-19, 'weighted_crb': 4.336808689942018e-19, 'target_sum_total': 4.336808689942018e-19, 'dwell_sum_s': 1.3877787807814457e-17, 'PD_formula_s': 6.938893903907228e-18, 'UD_formula_s': 0.0, 'FIM_relative': 6.380255061060816e-16}。外层参考诊断保留在outer_precision_checks.csv，没有替换主图AO结果。

基线点与(a)时域热启动的微小差别见baseline_point_checks.csv。本扫描所有点统一冷启动，所以不能强行使数值逐位匹配热启动轨迹；模型、预算和评价器完全相同。

## 文件与运行

run_sensitivity.py重算全部510次AO；plot_sensitivity.py只重绘本次CSV。PyCharm安装requirements.txt后直接运行run_sensitivity.py。完成目录受保护，重跑请复制代码、baseline_snapshot.json、allocation_methods.py及source_snapshot到新目录。

fig8b_five_methods和fig8c_five_methods分别有PNG/PDF/SVG及51行五方法CSV。method_points.csv、target_allocations.csv、controlled_states.csv、constraint_checks.csv、candidate_evaluations.csv及ao_trace.csv保存全部实际扫描状态、资源、FIM/CRB、约束、候选和更新过程。没有平滑、插值、固定偏移或删点。每张图5:4、600dpi，曲线顶到横轴两端。

## 图形验收

| 图片 | PDF字体下限pt | 碰撞 | 视觉 |
| --- | --- | --- | --- |
| fig8b_five_methods | 6.3 | PASS | PASS: actual final PNG inspected; all five raw curves and labels reviewed |
| fig8c_five_methods | 6.3 | PASS | PASS: actual final PNG inspected; all five raw curves and labels reviewed |
