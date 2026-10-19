"""Re-export the five reproduced figures from unmodified rerun CSV values."""
from pathlib import Path
import sys,csv,json,shutil,importlib.util
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'.deps').exists():sys.path.insert(0,str(ROOT/'.deps'))
sys.path.insert(0,str(ROOT/'legacy'))
import main as legacy
import matplotlib.pyplot as plt
RUN=ROOT.parent/'结果'/'legacy_reproduction'/'rerun';OUT=ROOT.parent/'图片'
def read(name):
    with (RUN/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
    OUT.mkdir(exist_ok=True)
    if not json.loads((ROOT.parent/'结果'/'legacy_reproduction'/'gate2_checks.json').read_text(encoding='utf-8'))['passed']:raise RuntimeError('Reproduction check must pass')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':10,'axes.labelsize':11,'legend.fontsize':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    original_save=legacy.save_figure
    def save(fig,path):
        fig.set_size_inches(6.25,5)
        for ax in fig.axes:
            if 'fig9' in path.stem:ax.set_ylabel('Target TTC-weighted wCRB')
            else:ax.set_ylabel('Total TTC-weighted wCRB')
            handles,labels=ax.get_legend_handles_labels()
            if 'fig9' in path.stem:labels=['k1','k2','k3','k4']
            if handles:
                old=ax.get_legend()
                if old is not None:old.remove()
                ax.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,1.09),ncol=min(len(labels),4),frameon=False,fontsize=9)
            for text in list(ax.texts):
                if text.get_text() in ['(a)','(b)','(c)']:
                    text.set_position((1,1.025));text.set_horizontalalignment('right');text.set_verticalalignment('bottom')
        fig.tight_layout()
        fig.canvas.draw()
        axes=fig.axes
        (OUT/(path.stem+'.layout.json')).write_text(json.dumps({'aspect':'5:4','canvas_inches':[6.25,5],
           'panels':len(axes),'alignment':'not applicable single panel','axes_rectangles':[list(ax.get_position().bounds) for ax in axes],
           'postprocessing_smoothing':False,'data_source':'fresh rerun CSV checked against history'},indent=2),encoding='utf-8')
        fig.savefig(path.with_suffix('.png'),dpi=600)
        fig.savefig(path.with_suffix('.pdf'))
        fig.savefig(path.with_suffix('.svg'))
        plt.close(fig)
    legacy.save_figure=save
    legacy.plot_multi_scenario_ao_convergence(read('fig7_multi_scenario_ao_trace.csv'),OUT/'fig7_multi_scenario_ao_convergence')
    legacy.plot_fig8a_time_wcrb(read('fig8a_time_wcrb.csv'),OUT/'fig8a_time_wcrb')
    legacy.plot_fig8_physical_sensitivity(read('fig8bc_physical_sensitivity.csv'),OUT/'fig8b_range_sensitivity',OUT/'fig8c_speed_sensitivity')
    legacy.plot_fig9_ttc_aware_target_wcrb(read('fig9_ttc_aware_target_wcrb.csv'),OUT/'fig9_ttc_aware_target_wcrb')
    legacy.save_figure=original_save
    for name in ['fig7_multi_scenario_ao_trace.csv','fig8a_time_wcrb.csv','fig8bc_physical_sensitivity.csv','fig9_ttc_aware_target_wcrb.csv']:
        shutil.copy2(RUN/name,OUT/name)
    print('Exported five PNG/PDF/SVG figures; raw point values unchanged')
if __name__=='__main__':main()
