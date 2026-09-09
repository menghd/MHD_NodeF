"""Read-only verification of a project's fixed MHD API source against its lock."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

def check(project_root, upstream=None):
    root=Path(project_root).resolve()
    lock=json.loads((root/'framework.lock.json').read_text())
    if lock['schema']!='mhd_source_lock_v1' or lock['api_version']!='V4':
        raise ValueError('Expected this preparation release to use fixed V4')
    results=[]
    for row in lock['files']:
        path=(root/row['local_path']).resolve()
        if not path.is_relative_to(root): raise ValueError('Source escapes project')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=row['sha256']: raise ValueError('Framework source drift: '+row['local_path'])
        if upstream:
            content=subprocess.check_output(['git','-C',str(upstream),'show',lock['upstream_commit']+':'+row['upstream_path']])
            if hashlib.sha256(content).hexdigest()!=digest: raise ValueError('Upstream source differs')
        results.append(row['local_path'])
    if lock['distribution']=='git_submodule':
        actual=subprocess.check_output(['git','-C',str(root/lock['submodule_path']),'rev-parse','HEAD'],text=True).strip()
        if actual!=lock['upstream_commit']: raise ValueError('Submodule commit differs')
    return dict(passed=True,api_version=lock['api_version'],upstream_commit=lock['upstream_commit'],files=results,upstream_verified=bool(upstream))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--upstream',type=Path)
    args=parser.parse_args()
    print(json.dumps(check(args.project_root,args.upstream),indent=2))
