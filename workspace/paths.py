"""Print portable project paths. Does not create directories or launch jobs."""
import argparse
import json
import re
from pathlib import Path

def resolve(config, machine, project, timestamp):
    if config['schema'] != 'souray_workspace_v1':
        raise ValueError('Unsupported workspace schema')
    if not re.fullmatch(r'\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}', timestamp):
        raise ValueError('Invalid timestamp')
    if project not in config['projects'] or machine not in config['machines']:
        raise ValueError('Unknown project or machine')
    roots = config['machines'][machine]
    if not all(Path(roots[k]).is_absolute() for k in ['code_root','data_root']):
        raise ValueError('Absolute roots required')
    values = dict(roots, project=project, timestamp=timestamp, project_lower=project.lower())
    result = {k: config[v].format(**values) for k,v in [
        ('code','source_pattern'),('run','run_pattern'),
        ('shared_dataset','shared_dataset_pattern'),('environment','environment_pattern')]}
    result['application'] = str(Path(result['code']) / config['projects'][project]['application_subdir'])
    result.update(project=project, timestamp=timestamp, machine=machine, creates_or_launches=False)
    return result

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry', type=Path, default=Path(__file__).with_name('registry.json'))
    p.add_argument('--machine', required=True)
    p.add_argument('--project', required=True)
    p.add_argument('--timestamp', required=True)
    args=p.parse_args()
    print(json.dumps(resolve(json.loads(args.registry.read_text()), args.machine, args.project, args.timestamp),indent=2))
