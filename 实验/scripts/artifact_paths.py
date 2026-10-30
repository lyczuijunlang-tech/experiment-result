"""Resolve removed byte-identical aliases without storing a second data copy."""
from pathlib import Path
import json
import shutil


def repository(path):
    for parent in Path(path).resolve().parents:
        if (parent / '结果' / 'deduplication.json').exists():
            return parent
    return None


def resolve_artifact(path):
    path = Path(path).resolve()
    if path.exists():
        return path
    root = repository(path)
    if root is None:
        return path
    index = json.loads((root / '结果/deduplication.json').read_text(encoding='utf-8'))
    aliases = index['removed_aliases']
    key = path.relative_to(root).as_posix()
    seen = set()
    while key in aliases:
        if key in seen:
            raise RuntimeError('Cyclic artifact alias: ' + key)
        seen.add(key)
        key = aliases[key]['canonical']
    return root / key


def figure_output(bundle, name):
    """Published bundles render into 图片; new independent runs stay local."""
    bundle = Path(bundle).resolve()
    root = repository(bundle / name)
    if root and bundle.relative_to(root).as_posix() in {
        '结果/fig8a_five_methods', '结果/fig8bc_five_methods'
    }:
        return root / '图片' / name
    return bundle / name


def materialize_directory(source, destination):
    """Restore frozen input aliases only into a NEW run's input directory."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source.exists():
        shutil.copytree(source, destination, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__'))
    root = repository(source / '_input_')
    if root is None:
        return
    prefix = source.relative_to(root).as_posix() + '/'
    aliases = json.loads((root / '结果/deduplication.json').read_text(encoding='utf-8'))['removed_aliases']
    for alias in aliases:
        if alias.startswith(prefix):
            target = destination / alias[len(prefix):]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolve_artifact(root / alias), target)
