"""Validate shared rules and optionally compare peer repository copies."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

FILES=('README.zh-CN.md','registry.json','paths.py','check.py','check_framework.py')
def check(root, peers=()):
    config=json.loads((root/'registry.json').read_text())
    spec=importlib.util.spec_from_file_location('workspace_paths',root/'paths.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert set(config['projects'])=={'LOOK','Radon_Bridge','MHD_Project'}
    assert config['raw_data_in_git'] is False and config['automatic_training_or_gpu_submission'] is False
    cases=0
    for machine in config['machines']:
        for project in config['projects']:
            result=module.resolve(config,machine,project,config['effective_from'])
            assert Path(result['code']).parts[-2:]==(project,config['effective_from'])
            assert Path(result['run']).parts[-3:]==(project,'runs',config['effective_from'])
            assert '/projects/' not in result['code']
            assert result['creates_or_launches'] is False
            cases+=1
    for value in ('../escape','', '2026_09_09_10_30_34/../../other'):
        try:module.resolve(config,'ibex','LOOK',value)
        except ValueError:pass
        else:raise AssertionError('unsafe path accepted')
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in FILES}
    for peer in peers:
        for name,sha in hashes.items():
            assert hashlib.sha256((peer/'workspace'/name).read_bytes()).hexdigest()==sha, f'{peer}: {name} differs'
    return dict(passed=True,path_cases=cases,peer_repositories=len(peers),sha256=hashes)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--peer',action='append',type=Path,default=[])
    args=p.parse_args();print(json.dumps(check(Path(__file__).resolve().parent,args.peer),indent=2))
