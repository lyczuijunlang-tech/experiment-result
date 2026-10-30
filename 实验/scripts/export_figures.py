"""Export current verified five-policy results; this is not a new simulation."""
from pathlib import Path
import argparse,csv,hashlib,json,shutil,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from artifact_paths import resolve_artifact
from deduplicate_artifacts import deduplicate

REPO=Path(__file__).resolve().parents[2]
RESULT=REPO/'结果';OUT=REPO/'图片'
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#5B5B5B']
STYLES=['-','--','-.',':',(0,(5,1,1,1))]
MARKERS=['o','s','^','D','v']
METHODS=['TTC-AO','RU-AO','DA-AO','TTC-PD','UD']
LABELS=['Proposed TTC-AO','Risk-Unaware AO','Distance-Aware AO','TTC-Proportional','Uniform-Dwell']
def read(path):
    with resolve_artifact(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def jsave(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def write_csv(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def copy(a,b):Path(b).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(resolve_artifact(a),b)
def alignment_gate(fig,path):
    helper=Path.home()/'.codex/skills/nature-figure/scripts'
    if helper.exists():
        sys.path.insert(0,str(helper))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        return require_matplotlib_panel_alignment(fig,json_out=path,tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    box=np.array([ax.get_position().bounds for ax in fig.axes])*np.array([fig.get_figwidth()*72,fig.get_figheight()*72]*2)
    gutters=box[1:,0]-box[:-1,0]-box[:-1,2]
    differences={'bottom':np.ptp(box[:,1]),'width':np.ptp(box[:,2]),'height':np.ptp(box[:,3]),'gutter':np.ptp(gutters)}
    assert all(v<=1.5 for v in differences.values()),differences
    jsave(path,{'auditor':'portable rendered-matplotlib equal-grid measurement','rectangles_pt':box.tolist(),
        'maximum_differences_pt':differences,'tolerance_pt':1.5,'verdict':'PASS'})

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fig7-dir',type=Path);parser.add_argument('--fig8-dir',type=Path);parser.add_argument('--fig8bc-dir',type=Path);args=parser.parse_args()
    release=json.loads((RESULT/'current_release.json').read_text(encoding='utf-8'))
    seven=args.fig7_dir or REPO/release['figure7_bundle'];five=args.fig8_dir or REPO/release['figure8a_bundle']
    bc=args.fig8bc_dir or (REPO/release['figure8bc_bundle'] if release.get('figure8bc_bundle') else None)
    if bc is not None:assert json.loads((bc/'completion.json').read_text())['all_checks_passed']
    assert json.loads((seven/'completion.json').read_text())['checks_passed']
    assert json.loads((five/'completion.json').read_text())['all_checks_passed']
    OUT.mkdir(exist_ok=True);qa=RESULT/'figure_qa';qa.mkdir(exist_ok=True)
    traj=RESULT/'trajectory_figures';traj.mkdir(exist_ok=True)
    wide=read(five/'fig8a_five_methods.csv');detail=[r for r in read(five/'resource_allocation_details.csv') if r['method']=='TTC-AO']
    states={(float(r['time_s']),r['vehicle_id']):r for r in read(five/'traffic.csv')}
    assert len(wide)==401 and len(detail)==1604
    distance=[];velocity=[];weighted=[]
    for r in detail:
        t=float(r['time_s']);ego=states[t,'ego'];vehicle=states[t,r['vehicle_id']]
        base={'cycle_index':r['cycle_index'],'time_s':t,'target_name':r['target'],'slot':r['slot'],'vehicle_id':r['vehicle_id']}
        distance.append({**base,'distance_m':float(r['distance_m']),'ego_x_m':float(ego['x_m']),'ego_y_m':float(ego['y_m']),
            'target_x_m':float(vehicle['x_m']),'target_y_m':float(vehicle['y_m'])})
        d=float(r['distance_m']);dx=float(vehicle['x_m'])-float(ego['x_m']);dy=float(vehicle['y_m'])-float(ego['y_m'])
        velocity.append({**base,'closing_speed_mps':float(r['closing_speed_mps']),
            'ego_vx_mps':float(ego['vx_mps']),'ego_vy_mps':float(ego['vy_mps']),
            'target_vx_mps':float(vehicle['vx_mps']),'target_vy_mps':float(vehicle['vy_mps']),
            'los_x':dx/d,'los_y':dy/d,'velocity_source':vehicle['velocity_source']})
        weighted.append({**base,'distance_m':d,'closing_speed_mps':float(r['closing_speed_mps']),
            'omega':float(r['physical_ttc_weight']),'dwell_s':float(r['dwell_s']),'Ps_w':float(r['Ps_w']),
            'CRB_d_m2':float(r['CRB_d_m2']),'CRB_v_m2ps2':float(r['CRB_v_m2ps2']),
            'scalar_CRB':float(r['scalar_crb']),'target_wcrb':float(r['final_ttc_weighted_crb']),
            'weighted_total_crb':float(wide[int(r['cycle_index'])]['TTC-AO'])})
    tables={'fig8b_target_distance':(distance,'distance_m','Relative distance (m)'),
            'fig8c_radial_velocity':(velocity,'closing_speed_mps','Radial closing speed (m/s)'),
            'fig9_ttc_aware_target_wcrb':(weighted,'target_wcrb','Target TTC-weighted wCRB')}
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':10,
        'axes.labelsize':11,'legend.fontsize':9,'axes.linewidth':.8,'xtick.direction':'in','ytick.direction':'in',
        'pdf.fonttype':42,'svg.fonttype':'none'})
    def draw_targets(ax,data,key):
        for i in range(4):
            rr=[r for r in data if r['target_name']==f'k{i+1}']
            ax.plot([r['time_s'] for r in rr],[r[key] for r in rr],color=COLORS[i],linestyle=STYLES[i],
                linewidth=1.6,marker=MARKERS[i],markevery=40,markersize=3,markerfacecolor='white',label=f'k{i+1}')
    def axis_style(ax,ylabel):
        ax.set(xlim=(0,40),xlabel='SUMO time (s)',ylabel=ylabel);ax.set_xticks(np.arange(0,41,10))
        ax.grid(color='.88',linewidth=.45);ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3),useMathText=True)
    for name,(data,key,ylabel) in tables.items():
        if name.startswith('fig9') and args.fig8_dir is None and release.get('current_export_source')==five.relative_to(REPO).as_posix() and (OUT/(name+'.png')).exists():continue
        plot_out=traj if bc is not None and not name.startswith('fig9') else OUT
        fig,ax=plt.subplots(figsize=(6.25,5.0));draw_targets(ax,data,key);axis_style(ax,ylabel)
        ax.legend(loc='lower center',bbox_to_anchor=(.5,1.09),ncol=4,frameon=False)
        fig.tight_layout();fig.canvas.draw()
        jsave(qa/(name+'.layout.json'),{'panel_count':1,'alignment_gate':'not applicable one panel','canvas_inches':[6.25,5.],
            'axes_position':list(ax.get_position().bounds),'data_source':five.relative_to(REPO).as_posix()+'/resource_allocation_details.csv',
            'smoothing':False,'time_samples_per_target':401})
        fig.savefig(plot_out/(name+'.png'),dpi=600);fig.savefig(plot_out/(name+'.pdf'));fig.savefig(plot_out/(name+'.svg'));plt.close(fig)
        write_csv(plot_out/(name+'.csv'),data)
        if plot_out==OUT:
            for ext in ['png','pdf','svg','csv']:
                copy(OUT/(name+'.'+ext),RESULT/(name+'.'+ext));copy(OUT/(name+'.'+ext),traj/(name+'.'+ext))
    sensitivity_rows=[]
    if bc is not None:
        for source in ['fig8b_five_methods','fig8c_five_methods']:
            sensitivity_rows.append(read(bc/(source+'.csv')))
            for ext in ['png','pdf','svg']:
                src=resolve_artifact(bc/(source+'.'+ext));dst=OUT/(source+'.'+ext)
                if src.resolve()!=dst.resolve():copy(src,dst)
    for name,src,source_name in [('fig7_multi_scenario_ao_convergence',seven,'fig7_convergence'),
            ('fig8a_five_methods',five,'fig8a_five_methods')]:
        for ext in ['png','pdf','svg']:
            source=resolve_artifact(src/(source_name+'.'+ext));destination=OUT/(name+'.'+ext)
            if source.resolve()!=destination.resolve():copy(source,destination)
    # Composite is intentionally wide; independent source exports remain 5:4.
    fig,axes=plt.subplots(1,3,figsize=(18.75,5.6))
    fig.subplots_adjust(left=.055,right=.985,bottom=.16,top=.72,wspace=.27)
    t=[float(r['time_s']) for r in wide]
    for i,(m,label) in enumerate(zip(METHODS,LABELS)):
        axes[0].plot(t,[float(r[m]) for r in wide],color=COLORS[i],linestyle=STYLES[i],linewidth=1.6,
            marker=MARKERS[i],markersize=3,markevery=(i*5,40),markerfacecolor='white',label=label)
    axis_style(axes[0],r'$\omega$CRB');axes[0].set_xlabel('t(s)')
    if bc is None:
        draw_targets(axes[1],distance,'distance_m');axis_style(axes[1],'Relative distance (m)')
        draw_targets(axes[2],velocity,'closing_speed_mps');axis_style(axes[2],'Radial closing speed (m/s)')
    else:
        for ax,rows,xlabel in zip(axes[1:],sensitivity_rows,['d(m)','v(m/s)']):
            x=[float(r['x_value']) for r in rows]
            for i,(m,label) in enumerate(zip(METHODS,LABELS)):
                ax.plot(x,[float(r[m]) for r in rows],color=COLORS[i],linestyle=STYLES[i],linewidth=1.6,
                    marker=MARKERS[i],markersize=3,markevery=(i,5),markerfacecolor='white',label=label)
            ax.set(xlim=(x[0],x[-1]),xlabel=xlabel,ylabel=r'$\omega$CRB')
            ax.grid(color='.88',linewidth=.45);ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,-3),useMathText=True)
    for i,ax in enumerate(axes):
        ax.text(1.,1.035,f'({chr(97+i)})',transform=ax.transAxes,ha='right',va='bottom',fontsize=10)
        if i==0 or bc is not None:
            h,l=ax.get_legend_handles_labels();order=[0,3,1,4,2]
            ax.legend([h[j] for j in order],[l[j] for j in order],loc='lower center',bbox_to_anchor=(.5,1.19),ncol=3,
                frameon=False,fontsize=8.2,columnspacing=.8,handlelength=2.2)
        else:ax.legend(loc='lower center',bbox_to_anchor=(.5,1.19),ncol=4,frameon=False,fontsize=9)
    fig.canvas.draw();alignment_gate(fig,qa/'fig8_abc_combined.alignment.json')
    fig.savefig(OUT/'fig8_abc_combined.png',dpi=600);fig.savefig(OUT/'fig8_abc_combined.pdf');fig.savefig(OUT/'fig8_abc_combined.svg');plt.close(fig)
    (traj/'difference_report.md').write_text('# 当前时域图导出依据\n\n图8(b)(c)和图9直接由当前五方法实际SUMO运行的TTC-AO目标记录和车辆状态导出。距离、径向闭合速度及逐目标wCRB全部使用原始逐周期值，未平滑或插值。\n\n每目标401个时间点，k1至k4编号与五方法资源明细一致。图9四条贡献之和等于TTC-AO系统总wCRB。历史独立重运行轨迹与当前五方法轨迹逐点一致，详见当前实验包baseline_comparison.json及independent_formula_checks.json。\n',encoding='utf-8')
    if bc is not None:
        (traj/'difference_report.md').write_text('# 辅助原始轨迹与图9\n\n本目录距离、速度图保留真实SUMO时域状态作为辅助资料，不再是当前主图8(b)(c)。当前主图b/c为五方法受控距离/闭合速度扫描，详见../fig8bc_five_methods/README.md。图9仍为TTC-AO四目标贡献时域图。所有辅助数据由实际五方法时域实验的同一车辆状态和目标记录导出。\n',encoding='utf-8')
        release['figure8bc_bundle']=bc.relative_to(REPO).as_posix()
    deduplicate()
    release['date']='2026-10-30'
    release['axis_labels']={'fig8a':{'x':'t(s)','y':'ωCRB'},'fig8b':{'x':'d(m)','y':'ωCRB'},'fig8c':{'x':'v(m/s)','y':'ωCRB'}}
    release['figure7_bundle']=seven.relative_to(REPO).as_posix();release['figure8a_bundle']=five.relative_to(REPO).as_posix()
    release['current_exports']={p.relative_to(REPO).as_posix():{'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in OUT.glob('*') if p.suffix in {'.png','.pdf','.svg','.csv'}}
    release['published_figures']=dict(release['current_exports'])
    release['current_export_source']=five.relative_to(REPO).as_posix();jsave(RESULT/'current_release.json',release)
    summary_path=RESULT/'run_summary.json'
    if summary_path.exists():
        summary=json.loads(summary_path.read_text(encoding='utf-8'));summary['current_release']=release;jsave(summary_path,summary)
    print('Exported current five figures and synchronized three-panel Fig8 assembly.')

if __name__=='__main__':main()
