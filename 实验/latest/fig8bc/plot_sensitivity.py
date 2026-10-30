"""Plot five actual strategies on controlled distance/closing-speed scans."""
from pathlib import Path

# Resolve the single stored copy after repository artifact deduplication.
import sys
for _parent in Path(__file__).resolve().parents:
    if (_parent/'实验/scripts/artifact_paths.py').exists():
        sys.path.insert(0,str(_parent/'实验/scripts'));break
try:
    from artifact_paths import resolve_artifact, figure_output, materialize_directory
except ModuleNotFoundError:
    def resolve_artifact(path):return Path(path)
    def figure_output(bundle,name):return Path(bundle)/name
    def materialize_directory(source,destination):
        import shutil
        shutil.copytree(source,destination,dirs_exist_ok=True)
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
METHODS=['TTC-AO','RU-AO','DA-AO','TTC-PD','UD']
LABELS=['Proposed TTC-AO','Risk-Unaware AO','Distance-Aware AO','TTC-Proportional','Uniform-Dwell']

def draw():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':10,
        'axes.labelsize':11,'legend.fontsize':8.3,'axes.linewidth':.8,'xtick.direction':'in','ytick.direction':'in',
        'pdf.fonttype':42,'svg.fonttype':'none'})
    colors=['#0072B2','#D55E00','#009E73','#CC79A7','#5B5B5B']
    styles=['-','--','-.',':',(0,(5,1,1,1))];markers=['o','s','^','D','v']
    qa=ROOT/'qa';qa.mkdir(exist_ok=True)
    for name,xlabel in [('fig8b_five_methods','d(m)'),('fig8c_five_methods','v(m/s)')]:
        with resolve_artifact(ROOT/(name+'.csv')).open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        x=np.array([float(r['x_value']) for r in rows]);assert len(rows)==51 and np.all(np.diff(x)>0)
        fig,ax=plt.subplots(figsize=(6.25,5.0))
        for i,(method,label) in enumerate(zip(METHODS,LABELS)):
            y=np.array([float(r[method]) for r in rows]);assert np.isfinite(y).all()
            ax.plot(x,y,color=colors[i],linestyle=styles[i],linewidth=1.7 if i==0 else 1.5,
                marker=markers[i],markevery=(i,5),markersize=3,markerfacecolor='white',markeredgewidth=.7,label=label)
        ax.set(xlim=(x[0],x[-1]),xlabel=xlabel,ylabel=r'$\omega$CRB')
        ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,-3),useMathText=True)
        ax.grid(color='.88',linewidth=.45)
        h,l=ax.get_legend_handles_labels();order=[0,3,1,4,2]
        ax.legend([h[i] for i in order],[l[i] for i in order],loc='lower center',bbox_to_anchor=(.5,1.09),ncol=3,
            frameon=False,columnspacing=.95,handlelength=2.4)
        fig.tight_layout();fig.canvas.draw()
        (qa/(name+'.layout.json')).write_text(json.dumps({'panel_count':1,'alignment_gate':'not applicable one panel',
            'canvas_inches':[6.25,5.],'axes_position':list(ax.get_position().bounds),'x_limits':list(ax.get_xlim()),
            'y_limits':list(ax.get_ylim()),'samples_per_method':51,'methods':5,'smoothing':False,
            'data_source':name+'.csv','experiment':'controlled single-SUMO-snapshot sensitivity'},indent=2),encoding='utf-8')
        fig.savefig(figure_output(ROOT,name+'.png'),dpi=600);fig.savefig(figure_output(ROOT,name+'.pdf'));fig.savefig(figure_output(ROOT,name+'.svg'));plt.close(fig)

if __name__=='__main__':draw()
