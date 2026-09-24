"""Verify committed source blobs; Git checkout EOL conversion is not an algorithm edit."""
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[2]
data = subprocess.check_output(['git', '-C', str(root), 'archive', '--format=zip', 'HEAD'])
with zipfile.ZipFile(io.BytesIO(data)) as archive:
    manifest = json.loads(archive.read('Exp/release/SOURCE_MANIFEST.json'))
    bad = []
    for item in manifest['retained_source_files']:
        try:
            observed = hashlib.sha256(archive.read(item['release_path'])).hexdigest()
        except KeyError:
            observed = None
        if observed != item['sha256']:
            bad.append(item['release_path'])
    if bad:
        raise SystemExit('Committed source mismatch: ' + ', '.join(bad))
changed = subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD']).returncode
if changed:
    raise SystemExit('Committed hashes pass, but tracked working files have local modifications.')
print('PASS: {} original source blobs match {}; tracked checkout clean (Git EOL normalization allowed).'.format(
    len(manifest['retained_source_files']), manifest['source_commit']))
