"""Recompute in a new timestamped folder, preserving previous experiment data."""
from pathlib import Path
from datetime import datetime
import argparse,shutil,subprocess,sys,os
ROOT=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dependency-dir',type=Path)
    args=parser.parse_args()
    dest=ROOT.parent/('fig8bc_five_methods_'+datetime.now().strftime('%Y%m%d_%H%M%S'));dest.mkdir()
    shutil.copytree(ROOT/'source_snapshot',dest/'source_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['baseline_snapshot.json','allocation_methods.py','run_sensitivity.py','plot_sensitivity.py','audit_and_report.py','prepare_snapshot.py','rerun.py','requirements.txt']:
        shutil.copy2(ROOT/name,dest/name)
    env=os.environ.copy()
    if args.dependency_dir:env['PYTHONPATH']=str(args.dependency_dir.resolve())+os.pathsep+env.get('PYTHONPATH','')
    subprocess.run([sys.executable,'-X','utf8',str(dest/'run_sensitivity.py')],check=True,env=env)
    if (Path.home()/'.codex/skills/nature-figure/scripts/audit_pdf_text.py').exists():
        subprocess.run([sys.executable,'-X','utf8',str(dest/'audit_and_report.py')],check=True,env=env)
    print('New actual sensitivity results:',dest)
if __name__=='__main__':main()
