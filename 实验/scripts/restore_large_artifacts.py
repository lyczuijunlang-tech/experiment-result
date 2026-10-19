"""Restore lossless NPZ parts after downloading the GitHub repository."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(1024**2):h.update(block)
    return h.hexdigest()
def main():
    count=0
    for manifest in (ROOT.parent/'结果').rglob('*.parts/manifest.json'):
        info=json.loads(manifest.read_text(encoding='utf-8'));name=info['original_name']
        if Path(name).name!=name:raise ValueError('Invalid output filename')
        target=manifest.parent.parent/name
        if target.exists():
            if digest(target)!=info['sha256']:raise RuntimeError('Existing file does not match manifest: '+str(target))
            print('Verified existing',target);continue
        temp=target.with_name(target.name+'.restoring')
        with temp.open('wb') as out:
            for part in info['parts']:
                if Path(part['name']).name!=part['name']:raise ValueError('Invalid part filename')
                p=manifest.parent/part['name']
                if p.stat().st_size!=part['bytes'] or digest(p)!=part['sha256']:raise RuntimeError('Part verification failed: '+str(p))
                with p.open('rb') as f:
                    while block:=f.read(1024**2):out.write(block)
        if temp.stat().st_size!=info['original_bytes'] or digest(temp)!=info['sha256']:raise RuntimeError('Restored file verification failed')
        temp.replace(target);count+=1;print('Restored',target)
    print('Newly restored artifacts',count)
if __name__=='__main__':main()
