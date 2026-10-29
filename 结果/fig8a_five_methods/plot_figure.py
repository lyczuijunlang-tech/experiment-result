"""Draw all 401 newly simulated time samples; no smoothing or artificial offsets."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
METHODS=('TTC-AO','RU-AO','DA-AO','TTC-PD','UD')
LABELS=('Proposed TTC-AO','Risk-Unaware AO','Distance-Aware AO','TTC-Proportional','Uniform-Dwell')

def draw():
    with (ROOT/'fig8a_five_methods.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    assert len(rows)==401
    time=np.array([float(r['time_s']) for r in rows])
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':10,
        'axes.labelsize':11,'legend.fontsize':8.3,'axes.linewidth':.8,'xtick.direction':'in','ytick.direction':'in',
        'pdf.fonttype':42,'svg.fonttype':'none'})
    styles=[('#0072B2','-','o'),('#D55E00','--','s'),('#009E73','-.','^'),('#CC79A7',':','D'),('#5B5B5B',(0,(5,1,1,1)),'v')]
    fig,ax=plt.subplots(figsize=(6.25,5.0))
    for i,(method,label) in enumerate(zip(METHODS,LABELS)):
        values=np.array([float(r[method]) for r in rows]);assert np.isfinite(values).all()
        color,style,marker=styles[i]
        ax.plot(time,values,color=color,linestyle=style,linewidth=1.7 if i==0 else 1.5,
            marker=marker,markevery=(i*5,40),markersize=3,markerfacecolor='white',markeredgewidth=.7,label=label)
    ax.set(xlim=(time[0],time[-1]),xlabel='Time (s)',ylabel='Total TTC-Weighted CRB')
    ax.set_xticks(np.arange(0,41,10));ax.grid(color='.88',linewidth=.45)
    ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3),useMathText=True)
    handles,labels=ax.get_legend_handles_labels();order=[0,3,1,4,2]
    ax.legend([handles[i] for i in order],[labels[i] for i in order],loc='lower center',bbox_to_anchor=(.5,1.09),
        ncol=3,frameon=False,columnspacing=.95,handlelength=2.4)
    fig.tight_layout();fig.canvas.draw()
    qa=ROOT/'qa';qa.mkdir(exist_ok=True)
    (qa/'layout.json').write_text(json.dumps({'panel_count':1,'alignment_gate':'not applicable single panel',
        'canvas_inches':[6.25,5.],'axes_position':list(ax.get_position().bounds),
        'x_limits':list(ax.get_xlim()),'y_limits':list(ax.get_ylim()),'all_time_points':401,
        'all_methods':5,'excluded_cycles':0,'smoothing':False,'uncertainty':'one nominal episode, no repeat-based band'},indent=2),encoding='utf-8')
    fig.savefig(ROOT/'fig8a_five_methods.png',dpi=600)
    fig.savefig(ROOT/'fig8a_five_methods.pdf')
    fig.savefig(ROOT/'fig8a_five_methods.svg')
    plt.close(fig)

if __name__=='__main__':draw()
