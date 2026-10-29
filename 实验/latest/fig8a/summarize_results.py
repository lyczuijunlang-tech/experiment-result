"""Summarize only the newly completed five-policy experiment."""
from pathlib import Path
import csv,json,math,shutil
import numpy as np

ROOT=Path(__file__).resolve().parent
METHODS=('TTC-AO','RU-AO','DA-AO','TTC-PD','UD')
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as file:return list(csv.DictReader(file))
def save(name,data):(ROOT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def table(headers,items):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(str(x) for x in r)+' |' for r in items]])
def csvsave(name,data):
    with (ROOT/name).open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)

def summarize():
    complete=json.loads((ROOT/'completion.json').read_text(encoding='utf-8'))
    assert complete['all_checks_passed']
    cycles=rows(ROOT/'method_cycles.csv');wide=rows(ROOT/'fig8a_five_methods.csv')
    details=rows(ROOT/'resource_allocation_details.csv');checks=rows(ROOT/'constraint_checks.csv')
    precision=rows(ROOT/'outer_precision_checks.csv');scale=rows(ROOT/'RU_scale_checks.csv')
    assert len(cycles)==len(checks)==2005 and len(wide)==401 and len(details)==8020
    means={m:float(np.mean([float(r['total_wcrb']) for r in cycles if r['method']==m])) for m in METHODS}
    statistics=[]
    for m in METHODS:
        rr=[r for r in cycles if r['method']==m];pr=[float(r['allocation_objective_gap_percent']) for r in precision if r['method']==m]
        statistics.append({'Method':m,'Mean wCRB':means[m],
            'Relative gain of TTC-AO (%)':'' if m=='TTC-AO' else 100*(means[m]-means['TTC-AO'])/means[m],
            'Feasible cycles':sum(int(r['final_feasible']) for r in rr),'Total cycles':len(rr),
            'Converged cycles':sum(int(r['ao_converged']) for r in rr),'Max-budget cycles':sum(1-int(r['ao_converged']) for r in rr),
            'Max outer objective gap (%)':max(pr),'Mean outer objective gap (%)':float(np.mean(pr)),
            'Complete P1 solves':sum(int(r['complete_p1_solves']) for r in rr),
            'Direct allocations':sum(int(r['direct_allocations']) for r in rr)})
    csvsave('method_statistics.csv',statistics)
    spreads=[{'cycle_index':i,'time_s':wide[i]['time_s'],
        'Ts_max_minus_min_s':float(np.ptp([float(r['Ts_s']) for r in cycles if int(r['cycle_index'])==i])),
        'Ps_max_minus_min_w':float(np.ptp([float(r['Ps_w']) for r in cycles if int(r['cycle_index'])==i]))}
        for i in range(401)]
    csvsave('system_resource_comparison.csv',spreads)
    resource_spread={key:max(r[key] for r in spreads) for key in ['Ts_max_minus_min_s','Ps_max_minus_min_w']}
    target_summary=[]
    for method in METHODS:
        for k in ('k1','k2','k3','k4'):
            rr=[r for r in details if r['method']==method and r['target']==k]
            target_summary.append({'method':method,'target':k,**{key:float(np.mean([float(r[key]) for r in rr]))
                for key in ['distance_m','closing_speed_mps','physical_ttc_weight','optimization_weight','dwell_s','final_ttc_weighted_crb']}})
    csvsave('target_allocation_summary.csv',target_summary)
    pairwise=[]
    for i,a in enumerate(METHODS):
        for b in METHODS[i+1:]:
            av=np.array([float(r[a]) for r in wide]);bv=np.array([float(r[b]) for r in wide]);delta=av-bv
            nonzero=delta[np.abs(delta)>1e-12];cross=int(np.sum(np.sign(nonzero[1:])!=np.sign(nonzero[:-1])))
            pairwise.append({'method_A':a,'method_B':b,'sign_crossings':cross,
                'mean_relative_absolute_gap_percent':float(np.mean(np.abs(delta)/np.maximum(av,bv))*100),
                'B_better_than_A_cycles':int(np.sum(delta>1e-12)),
                'near_equal_within_1e_12_cycles':int(np.sum(np.abs(delta)<=1e-12))})
    csvsave('pairwise_curve_checks.csv',pairwise)
    # Historical CSV access occurs only after actual SUMO has completed.
    historical=ROOT/'reference_baseline'
    if not historical.exists():
        previous=ROOT.parent/'fig7_fig8_fig9_rerun/raw_run'
        historical.mkdir()
        for name in ['cycles.csv','traffic.csv','fig8a_time_wcrb.csv']:
            shutil.copy2(previous/name,historical/name)
    oldcycles=rows(historical/'cycles.csv');newttc=[r for r in cycles if r['method']=='TTC-AO']
    errors={new: max(abs(float(a[old])-float(b[new])) for a,b in zip(oldcycles,newttc))
        for old,new in [('weighted_total_crb','total_wcrb'),('Ts_star_s','Ts_s'),('Ps_star_w','Ps_w'),('AO_iterations','ao_rounds')]}
    oldtraffic=rows(historical/'traffic.csv');traffic=rows(ROOT/'traffic.csv')
    def keyed(data):return {(float(r['time_s']),r['vehicle_id']):r for r in data}
    a,b=keyed(oldtraffic),keyed(traffic);assert set(a)==set(b)
    state_errors={key:max(abs(float(a[index][key])-float(b[index][key])) for index in a) for key in ['x_m','y_m','vx_mps','vy_mps','speed_mps','heading_rad','lane_index']}
    assert max(errors.values())<1e-12 and max(state_errors.values())<1e-12
    oldablation=rows(historical/'fig8a_time_wcrb.csv');newru=[r for r in cycles if r['method']=='RU-AO']
    no_ttc_change=max(abs(float(a['no_ttc_total_wcrb'])-float(b['total_wcrb'])) for a,b in zip(oldablation,newru))
    baseline={'historical_data_opened_after_fresh_SUMO_completion':True,'TTC_original_max_absolute_errors':errors,
        'nominal_trajectory_max_absolute_errors':state_errors,'old_No_TTC_vs_independent_RU_P2_max_wcrb_change':no_ttc_change,
        'old_No_TTC_scope':'P1-only at TTC-AO system resources, optimization weights 1/K',
        'new_RU_scope':'Own complete P2 with weights 1; same feasible set and final physical TTC evaluation',
        'RU_scale_probe_max_errors':{key:max(abs(float(r[key])) for r in scale) for key in ['max_dwell_difference_s','Ts_difference_s','Ps_difference_w','common_wcrb_difference','ao_round_difference']}}
    save('baseline_comparison.json',baseline)
    assert max(baseline['RU_scale_probe_max_errors'].values())<1e-12
    minimum_keys=['minimum_dwell_margin_s','time_lower_margin_s','time_upper_margin_s','energy_margin_j',
                  'power_lower_margin_w','power_upper_margin_w','qos_rate_margin_bps']
    minima={m:{key:min(float(r[key]) for r in checks if r['method']==m) for key in minimum_keys} for m in METHODS}
    bycycle={i:[] for i in range(401)}
    for r in details:bycycle[int(r['cycle_index'])].append(r)
    per_method=ROOT/'per_method';per_method.mkdir(exist_ok=True)
    for method in METHODS:
        packed=[]
        for row in cycles:
            if row['method']!=method:continue
            item=dict(row)
            for target in bycycle[int(row['cycle_index'])]:
                if target['method']!=method:continue
                k=target['target']
                for key in ['physical_ttc_weight','optimization_weight','scalar_crb','final_ttc_weighted_crb','CRB_d_m2','CRB_v_m2ps2']:
                    item[k+'_'+key]=target[key]
            packed.append(item)
        csvsave('per_method/'+method+'.csv',packed)
    for i,rr in bycycle.items():
        for k in ('k1','k2','k3','k4'):
            target=[r for r in rr if r['target']==k]
            assert len(target)==5 and len({r['physical_ttc_weight'] for r in target})==1
        assert len({r['state_sha256'] for r in rr})==1
    audit={'passed':True,'all_401_cycles_included':True,'all_2005_method_cycles_feasible':all(int(r['feasible']) for r in checks),
        'all_final_CRBs_and_FIMs_finite':all(int(r['all_final_values_finite']) and float(r['final_fim_min_eigenvalue'])>0 for r in checks),
        'identical_physical_TTC_weights_for_all_methods_each_cycle':True,'identical_state_for_all_methods_each_cycle':True,
        'max_dwell_sum_error_s':max(float(r['dwell_sum_error_s']) for r in checks),
        'max_target_sum_vs_total_error':max(float(r['target_sum_vs_total_error']) for r in checks),
        'minimum_constraint_margins_by_method':minima,
        'full_AO_code_object_reused_for_all_methods':True,'five_methods_same_initialization_budget_tolerances':True,
        'independently_solved_system_resource_max_spread':resource_spread,
        'direct_policy_KKT':'not applicable, blank; direct allocation is not falsely reported as an optimal P1 solve',
        'zero_power_endpoints':'Expected original singular power boundary recorded as text; no nonfinite final result',
        'no_future_state_or_teacher_input':True,'no_curve_smoothing_or_trajectory_perturbation':True}
    assert audit['all_2005_method_cycles_feasible'] and audit['all_final_CRBs_and_FIMs_finite']
    save('scientific_audit.json',audit)
    rows_stats=[[s['Method'],f"{s['Mean wCRB']:.12g}",'—' if s['Method']=='TTC-AO' else f"{s['Relative gain of TTC-AO (%)']:.6f}",s['Feasible cycles']] for s in statistics]
    precision_table=[[s['Method'],s['Converged cycles'],s['Max-budget cycles'],f"{s['Mean outer objective gap (%)']:.8g}",f"{s['Max outer objective gap (%)']:.8g}"] for s in statistics]
    crossover_table=[[r['method_A']+' / '+r['method_B'],r['sign_crossings'],f"{r['mean_relative_absolute_gap_percent']:.5f}",r['B_better_than_A_cycles']] for r in pairwise]
    report=['# Fig. 8(a) 五方法实际运行审查报告',
        f"实际新运行：{complete['started_at_utc']} 至 {complete['completed_at_utc']} UTC；耗时{complete['wall_seconds']:.3f}秒。单次原始确定性SUMO40秒、401周期，五方法2005组最终资源，8020条目标级结果。",
        '## 方法实现与共同评价',
        table(['方法','目标级分配','P2优化的目标'],[
            ['TTC-AO','原完整KKT P1，当前TTC风险权重','ΣωTTC Ck'],
            ['RU-AO','原完整KKT P1，权重1','ΣCk'],
            ['DA-AO','原完整KKT P1，χ/(d+0.1m)归一到和K','ΣωDA Ck'],
            ['TTC-PD','Tmin+(Ts−K Tmin)ωTTC/ΣωTTC，直接计算','ΣωTTC Ck'],
            ['UD','Tk=Ts/K，直接计算','ΣωTTC Ck']]),
        '五者分别完整调用原始AO系统层，使用同一初始Ts=25ms、Ps=13.5dBm、η=1e-6、最多100轮、时间松弛0.35、功率松弛0.03/接近时间上界0.15、跨周期仅使用各自上一周期系统资源及原0.995回退。所有最终曲线统一调用evaluate_common_ttc_wcrb，以同周期真实物理TTC权重评价。',
        'DA资格条件与原TTC一致：闭合速度v>0；除资格条件外，距离优先级不使用闭合速度大小。全零风险按原方法回退为等权。RU-AO仍优化各目标驻留，不能与UD混称。PD/UD的接口兼容计数只称inner evaluations，实际完整P1求解次数为0；不把直接分配的KKT残差虚报为0。',
        '## 全时域均值与改善比例',
        table(['Method','Mean wCRB','Relative gain of TTC-AO (%)','Feasible cycles'],rows_stats),
        '改善比例严格按(基线全时域均值−TTC-AO全时域均值)/基线全时域均值×100%计算；不是逐周期百分比的平均。负值若出现会原样保留。',
        '## 外层精度与公平性',
        table(['Method','早停周期','达到最大轮数周期','平均自身目标差距 %','最大自身目标差距 %'],precision_table),
        '这些差距的分母使用每个方法自身的分配目标，不能直接当成共同TTC指标的方法间改善。P2参数对五者完全相同；上表区分满足早停与耗尽预算，不能将达到100轮写成早停收敛。',
        f'每周期独立求解得到的五方法系统资源最大差：{resource_spread}。本次数据中Ts/Ps恰好相同，这与本模型的单调性及共同边界一致，不表示复用了TTC-AO解。全部方法分别记录了独立候选调用。因此本数据中的wCRB差异由目标级驻留规则与优先级信息产生；它不支持“系统资源值有明显差异”的结论。',
        '对于TTC/RU/DA的最优内层及UD，在固定状态和本模型下，功率取其可行上界最优。沿能量边界扩大Ts并同比扩大可行驻留，距离项不增、多普勒项下降，最低驻留约束仍满足；最大QoS允许Ts与其功率上界提供独立模型缩放证书。PD带固定Tmin偏置的占比随Ts变化，因此额外用81点可行时间网格加局部有界细化检查，不宣称PD参考值获得了严格全局证书。参考结果仅作诊断，没有替换主图的原AO解。',
        '## 可行性与数值检查',
        f"全部2005组最终资源可行，最终CRB/FIM有限且FIM正定；最大ΣTk−Ts误差{audit['max_dwell_sum_error_s']:.6g}s，最大目标贡献和误差{audit['max_target_sum_vs_total_error']:.6g}。全部方法每周期共享状态哈希及物理TTC权重，原始非有限零功率端点被明确标为不可辨识候选，不作为最终有限估计。",
        '逐周期约束裕量见constraint_checks.csv，汇总见scientific_audit.json；完整候选和AO更新见candidate_evaluations.csv、ao_trace.csv。',
        '## 原实现一致性与新旧差异',
        f"TTC-AO与历史基准逐点最大差：{errors}；实际轨迹逐字段最大差：{state_errors}。",
        f"RU权重1与1/K在五个固定审查周期的资源、驻留、共同评价和轮数最大差：{baseline['RU_scale_probe_max_errors']}。两者的目标尺度相差K，理论最优相同；本次相对停止准则及归一化KKT数值尺度检查已单独记录。",
        f"历史No-TTC只在TTC系统资源处重新分配P1，本次RU完整优化自己的P2；两版No-TTC最大wCRB差为{no_ttc_change:.12g}，相近或一致并不表示复用了旧资源解。历史CSV在新SUMO完成后才读取作差异核对。",
        '本次按任务书返回未扰动的原始SUMO场景，不沿用上一张30次受控扰动均值±标准差图。只有一条原始动态轨迹，故本图不绘制无法从独立重复估计的鲁棒性阴影。',
        '## 重叠、交叉与解释边界',
        table(['方法对','符号交叉次数','平均相对绝对差 %','B优于A的周期数'],crossover_table),
        '该表平均相对绝对差仅用于检查曲线接近程度，不是正文定义的两个均值之比Gain。所有接近、交叉或顺序均由实际数据保留。不同权重之间的差异反映信息与优先级选择；TTC-AO与TTC-PD在相同风险信息下的差异用于观察CRB边际优化相较风险比例分配的作用；RU-AO与UD用于观察目标级时间自适应的作用。各自P2已重新优化，本次实际系统资源相同，因此曲线差异可定位到目标级规则；仍不能把总增益进一步拆成可加的跨场景因果百分比。',
        '数据支持本路网、本预算、本参数和当前EFIM下的有限时域比较；不直接推广为其他交通、估计误差、完整FIM或TTC精度约束下的优势。没有预先限定方法排名，也没有根据排名调整交通、预算或求解器参数。',
        '## 代码与复现',
        '新增allocation_methods.py、run_five_methods.py、plot_figure.py、summarize_results.py与audit_results.py。source_snapshot中的原模型、原主程序和SUMO文件逐字节保留。原单基地雷达、匹配波束、LFM理论回波、未知复幅度消元后的对角EFIM、无量纲CRB和全部资源约束不变。',
        '命令：python run_five_methods.py --sumo-home "C:\\Program Files (x86)\\Eclipse\\Sumo"。已完成目录受保护；如需重跑，先复制代码、requirements.txt与source_snapshot到新的独立目录。PyCharm使用安装requirements.txt的Python解释器即可。绘图可单独运行python plot_figure.py，后者只重绘本次新CSV，不代表重新仿真。']
    (ROOT/'experiment_report.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    readme='# Fig. 8(a) 五种真实资源分配方法\n\n'+table(['Method','Mean wCRB','TTC-AO relative gain (%)','Feasible cycles'],rows_stats)+'\n\n'
    readme+='详细实现、公式、相同预算和共同评价口径、外层精度、原实现差异及结论边界见[实验报告](experiment_report.md)。\n\n'
    readme+='运行：`python run_five_methods.py --sumo-home "C:\\Program Files (x86)\\Eclipse\\Sumo"`。PyCharm先安装requirements.txt。完成目录受保护，重新计算请复制代码与source_snapshot到新目录。\n\n'
    readme+='推荐直接运行`rerun.py`：自动复制源代码及只用于事后对比的reference_baseline，在新时间戳目录重新执行SUMO和全部五方法。数学公式、变量与单位见[公式说明](formula_and_variables.md)。\n\n'
    readme+='五张方法曲线共同位于fig8a_five_methods.png/.pdf/.svg；401行共同指标见fig8a_five_methods.csv；各方法逐周期系统资源见method_cycles.csv；8020条目标级驻留、优化权重、共同TTC权重、CRB及FIM见resource_allocation_details.csv；统计见method_statistics.csv；可行性见constraint_checks.csv；原始SUMO轨迹见traffic.csv及warmup_traffic.csv；每个候选与AO更新见candidate_evaluations.csv和ao_trace.csv。\n\n'
    readme+='`per_method/`另含五个独立CSV，各自401行，合并该方法的系统资源、各目标驻留、未加权及共同TTC加权CRB、权重、求解器状态，便于单独检查。完整原始长表也保留。\n\n'
    readme+='本版是单次原始确定性SUMO场景的五种完整P2策略比较，没有30次扰动阴影，没有更换完整FIM或加入TTC精度约束。没有平滑、加偏移、删除周期或预设排序。图7、8(b)(c)、9及历史结果未覆盖。\n'
    (ROOT/'README.md').write_text(readme,encoding='utf-8')
    print(table(['Method','Mean wCRB','Gain %','Feasible cycles'],rows_stats),flush=True)

if __name__=='__main__':summarize()
