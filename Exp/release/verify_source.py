import json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[2]
m=json.loads((root/'Exp/release/SOURCE_MANIFEST.json').read_text())
bad=[]
for f in m['retained_source_files']:
 p=root/f['release_path']
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=f['sha256']:bad.append(f['release_path'])
if bad:raise SystemExit('Source hash mismatch: '+', '.join(bad))
print('PASS: '+str(len(m['retained_source_files']))+' retained source files match original commit '+m['source_commit'])
