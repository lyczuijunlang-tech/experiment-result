# 数学公式与实际代码映射

| 公式及过程 | 代码位置 |
| --- | --- |
| d = //p_k − p_0//₂；v = −(V_k−V_0)ᵀ(p_k−p_0)/d；φ为本车坐标系方位 | source_snapshot/model.py:209 (relative_observation) |
| TTC=d/v 当v>0，否则∞；raw=1/(TTC+0.1)；ω=K raw/Σraw，全零则ω=1 | source_snapshot/model.py:276 (compute_ttc_weights) |
| w=a_t///a_t//；b=a_r///a_r//；每周期重算 | source_snapshot/model.py:311 (closed_form_beams) |
| Pr = Ps Gt Gr λ² σ / [(4π)³ d⁴ L] | source_snapshot/model.py:331 (radar_received_power_w) |
| Jdd=8π²(Pr/N0)(2/c)²(B²/12)Tk；Jvv=8π²(Pr/N0)(2fc/c)²Tk³/12；Jdv=0 | source_snapshot/model.py:410 (fim_and_crb) |
| Ck = CRBd/(1 m²) + CRBv/[1 (m/s)²]；内层目标ΣωCk | source_snapshot/model.py:540 (solve_p1) |
| KKT对偶二分100次；每目标Tk≥Tmin，ΣTk=Ts | source_snapshot/model.py:474 (_allocate_convex_dwell) |
| Tc,min = T Rmin/[Bc log₂(1+γc)]；Ts≤T−Tc,min；Ps Ts≤Es,max | source_snapshot/model.py:601 (constraint_margins) |
| 每个外层候选先查可行性，并完整调用新的P1，无缓存 | source_snapshot/model.py:630 (CandidateEvaluator) |
| 固定Ps优化Ts，再固定新Ts优化Ps；保留欠松弛 | source_snapshot/model.py:773 (alternating_optimize_system) |
