"""Small CPU-only library probe; does not change any running worker."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np

OUT = Path('/nvme/khy/ForesightNav_exp4_depthfix/results/exp4')
rng = np.random.RandomState(20261109)
cases = []
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
for height, width in [(240, 240), (480, 640), (1024, 1024)]:
    rgb = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
    mask = (rng.rand(height, width) > .83).astype(np.uint8)
    cases += [(f'rgb_to_bgr_{height}x{width}', lambda rgb=rgb: cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)),
              (f'dilate_{height}x{width}', lambda mask=mask: cv2.dilate(mask, kernel, iterations=2)),
              (f'connected4_{height}x{width}', lambda mask=mask: cv2.connectedComponents(mask, connectivity=4))]
initial = cv2.getNumThreads()
results, reference = [], {}
for mode, threads in [('default', initial), ('single', 1)]:
    cv2.setNumThreads(threads)
    for name, operation in cases:
        output = operation()
        samples = []
        for _ in range(3):
            start = time.perf_counter()
            output = operation()
            samples.append((time.perf_counter() - start) * 1000.)
        normalized = output if isinstance(output, tuple) else (output,)
        encoded = [(np.asarray(v).shape, str(np.asarray(v).dtype),
                    hashlib.sha256(np.asarray(v).tobytes()).hexdigest()) for v in normalized]
        if mode == 'default':
            reference[name] = tuple(np.asarray(v).copy() for v in normalized)
            identical = True
        else:
            identical = len(normalized) == len(reference[name]) and all(
                np.array_equal(np.asarray(a), b) for a, b in zip(normalized, reference[name]))
        results.append({'mode': mode, 'threads': cv2.getNumThreads(), 'operation': name,
                        'median_ms': statistics.median(samples), 'samples_ms': samples,
                        'output_digest': encoded, 'bitwise_identical_to_default': identical})
value = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
         'scope': 'fixed synthetic CPU library calls only; not native navigation throughput or score',
         'opencv_version': cv2.__version__, 'original_default_threads': initial,
         'all_outputs_bitwise_identical': all(r['bitwise_identical_to_default'] for r in results),
         'results': results, 'repeats': 3, 'random_seed': 20261109,
         'running_worker_environment_changed': False, 'runtime_source_changed': False,
         'parameters_fitted': False, 'native_workers_started': False,
         'interpretation': 'Exploratory timing under changing shared host load; not a formal speedup estimate. Do not apply during frozen calibration.'}
path = OUT / 'diagnostics/opencv-threads-synthetic.json'
path.write_text(json.dumps(value, indent=2) + '\n')
print(json.dumps({'opencv_version': value['opencv_version'], 'default_threads': initial,
                  'all_outputs_bitwise_identical': value['all_outputs_bitwise_identical'],
                  'timings': [{'mode': r['mode'], 'operation': r['operation'],
                               'median_ms': r['median_ms']} for r in results]}, indent=2))

