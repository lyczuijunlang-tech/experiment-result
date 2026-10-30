"""One-click new independent experiment directory; historical output is protected."""
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
from datetime import datetime
import argparse,shutil,subprocess,sys,os

def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sumo-home',type=Path,default=Path(r'C:\Program Files (x86)\Eclipse\Sumo'))
    parser.add_argument('--dependency-dir',type=Path)
    args=parser.parse_args()
    destination=root.parent/('fig8a_five_methods_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    destination.mkdir()
    for name in ['source_snapshot','reference_baseline']:
        materialize_directory(root/name,destination/name)
    for name in ['allocation_methods.py','run_five_methods.py','summarize_results.py','plot_figure.py','audit_results.py','rerun.py','requirements.txt','formula_and_variables.md','source_reference.json']:
        shutil.copy2(root/name,destination/name)
    env=os.environ.copy()
    if args.dependency_dir:env['PYTHONPATH']=str(args.dependency_dir.resolve())+os.pathsep+env.get('PYTHONPATH','')
    subprocess.run([sys.executable,'-X','utf8',str(destination/'run_five_methods.py'),'--sumo-home',str(args.sumo_home.resolve())],check=True,env=env)
    if (Path.home()/'.codex/skills/nature-figure/scripts/audit_pdf_text.py').exists():
        subprocess.run([sys.executable,'-X','utf8',str(destination/'audit_results.py')],check=True,env=env)
    print('New independent result:',destination)

if __name__=='__main__':main()
