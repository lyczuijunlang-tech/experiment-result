"""Freeze a reference from an actually completed Fig8(a) run, not its images."""
from pathlib import Path
import argparse,csv,json,hashlib
ROOT=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fig8a-dir',type=Path,required=True);args=parser.parse_args();base=args.fig8a_dir.resolve()
    def rows(name):
        with (base/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    assert json.loads((base/'completion.json').read_text())['all_checks_passed']
    grouped={}
    for r in rows('resource_allocation_details.csv'):
        if r['method']=='TTC-AO':grouped.setdefault(int(r['cycle_index']),[]).append(r)
    i=max(grouped,key=lambda j:max(float(r['physical_ttc_weight']) for r in grouped[j])-min(float(r['physical_ttc_weight']) for r in grouped[j]))
    methods=[r for r in rows('method_cycles.csv') if int(r['cycle_index'])==i]
    cycle=next(r for r in methods if r['method']=='TTC-AO');t=float(cycle['time_s'])
    config=json.loads((base/'run_config.json').read_text(encoding='utf-8'))
    config['system']['communication_snr_linear']=float(cycle['communication_snr_linear'])
    frozen={'cycle_index':i,'time_s':t,'vehicle_states':[r for r in rows('traffic.csv') if float(r['time_s'])==t],
        'radar':config['radar'],'system':config['system'],'baseline_targets':grouped[i],'baseline_all_methods':methods,
        'baseline_selection':'original offline maximum physical-risk-spread snapshot; not online future-state input',
        'source_csv_sha256':{n:hashlib.sha256((base/n).read_bytes()).hexdigest() for n in ['traffic.csv','method_cycles.csv','resource_allocation_details.csv']},
        'immutable_physical_source_sha256':config['source_sha256']}
    (ROOT/'baseline_snapshot.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Recorded reference frozen:',i,t)
if __name__=='__main__':main()
