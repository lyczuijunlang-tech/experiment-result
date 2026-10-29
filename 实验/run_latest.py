"""Run current experiments into a new directory and update published figures."""
from pathlib import Path
from datetime import datetime
import argparse,shutil,subprocess,sys
CODE=Path(__file__).resolve().parent
REPO=CODE.parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sumo-home',type=Path,default=Path(r'C:\Program Files (x86)\Eclipse\Sumo'))
    args=parser.parse_args()
    dest=REPO/'结果/latest_runs'/datetime.now().strftime('%Y%m%d_%H%M%S');dest.mkdir(parents=True)
    for name in ['fig7','fig8a','fig8bc']:shutil.copytree(CODE/'latest'/name,dest/name,ignore=shutil.ignore_patterns('__pycache__'))
    subprocess.run([sys.executable,'-X','utf8',str(dest/'fig7/run_fig7.py')],check=True)
    subprocess.run([sys.executable,'-X','utf8',str(dest/'fig8a/run_five_methods.py'),'--sumo-home',str(args.sumo_home.resolve())],check=True)
    subprocess.run([sys.executable,'-X','utf8',str(dest/'fig8bc/prepare_snapshot.py'),'--fig8a-dir',str(dest/'fig8a')],check=True)
    subprocess.run([sys.executable,'-X','utf8',str(dest/'fig8bc/run_sensitivity.py')],check=True)
    subprocess.run([sys.executable,'-X','utf8',str(CODE/'scripts/export_figures.py'),'--fig7-dir',str(dest/'fig7'),'--fig8-dir',str(dest/'fig8a'),'--fig8bc-dir',str(dest/'fig8bc')],check=True)
    print('New actual run:',dest)

if __name__=='__main__':main()
