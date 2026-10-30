"""Remove only SHA-256-identical image/data copies; retain a resolvable index."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json

REPO = Path(__file__).resolve().parents[2]
SUFFIXES = {'.png', '.pdf', '.svg', '.csv', '.npz', '.npy', '.xlsx'}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rank(path):
    name = path.relative_to(REPO).as_posix()
    if path.suffix == '.json':
        return (0 if name.startswith('结果/') else 1, name)
    if path.suffix in {'.png', '.pdf', '.svg'}:
        preferred = name.startswith('图片/') and not any(
            token in name for token in ('time_wcrb', 'range_sensitivity', 'target_distance',
                                        'speed_sensitivity', 'radial_velocity'))
        return (0 if preferred else 1 if name.startswith('图片/') else 2, name)
    if name.startswith(('结果/fig7_latest/', '结果/fig8a_five_methods/',
                        '结果/fig8bc_five_methods/')) and '/inputs/' not in name and '/reference_baseline/' not in name:
        priority = 0
    elif name.startswith('实验/latest/'):
        priority = 1
    elif name.startswith('实验/reference/legacy_results/'):
        priority = 2
    elif name.startswith('结果/trajectory_figures/'):
        priority = 3
    elif name.startswith('结果/legacy_reproduction/rerun/'):
        priority = 4
    else:
        priority = 5
    return priority, name


def deduplicate():
    index_path = REPO / '结果/deduplication.json'
    index = json.loads(index_path.read_text(encoding='utf-8')) if index_path.exists() else {
        'policy': 'Only byte-identical SHA-256 image/data/layout copies are removed. PNG/PDF/SVG are distinct required formats.',
        'removed_aliases': {},
    }
    groups = defaultdict(list)
    for path in REPO.rglob('*'):
        if any(part in {'.git', '__pycache__', '.deps'} for part in path.relative_to(REPO).parts):
            continue
        if path.is_file() and (path.suffix.lower() in SUFFIXES or path.name.endswith('.layout.json') or path.name == 'layout.json'):
            groups[digest(path)].append(path)
    count = saved = 0
    for checksum, paths in groups.items():
        if len(paths) < 2:
            continue
        canonical = min(paths, key=rank)
        for path in paths:
            if path == canonical:
                continue
            # Each resolved target is checked inside this explicit repository.
            assert path.resolve().is_relative_to(REPO.resolve())
            assert digest(path) == digest(canonical) == checksum
            size = path.stat().st_size
            index['removed_aliases'][path.relative_to(REPO).as_posix()] = {
                'canonical': canonical.relative_to(REPO).as_posix(),
                'original_sha256': checksum, 'bytes_removed': size,
            }
            path.unlink()
            count += 1
            saved += size
    # Flatten aliases before a later label-only redraw changes image bytes.
    aliases = index['removed_aliases']
    for old, entry in aliases.items():
        seen = {old}
        while entry['canonical'] in aliases:
            key = entry['canonical']
            assert key not in seen
            seen.add(key)
            entry['canonical'] = aliases[key]['canonical']
        assert (REPO / entry['canonical']).is_file(), entry
    index.update(updated_utc=datetime.now(timezone.utc).isoformat(),
                 removed_file_count=len(aliases),
                 total_duplicate_bytes_removed=sum(r['bytes_removed'] for r in aliases.values()))
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Duplicate copies removed: {count}; bytes removed: {saved}')
    return index


if __name__ == '__main__':
    deduplicate()
