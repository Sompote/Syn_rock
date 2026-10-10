"""Blender smoke test: list the Cycles GPU devices, then time two full-resolution foundation faces.

  blender -b --factory-startup -P blender/smoke_test.py
The second face shows the steady-state time, after the GPU kernels have loaded.
"""
import json, os, sys, tempfile, time
import bpy

prefs = bpy.context.preferences.addons['cycles'].preferences
found = {}
for backend in ('OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI'):
    try:
        prefs.compute_device_type = backend
    except TypeError:
        continue
    prefs.get_devices()
    names = [d.name for d in prefs.devices if d.type == backend]
    if names:
        found[backend] = names
print('BLENDER', bpy.app.version_string)
print('GPU BACKENDS', json.dumps(found))
if not found:
    print('WARNING: no GPU backend; check the NVIDIA driver and that this Blender build supports the card')

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)
from tunnelgen.sampler import sample_job  # noqa: E402
tmp = tempfile.mkdtemp()
json.dump([sample_job(900001), sample_job(900002)], open(os.path.join(tmp, 'jobs.json'), 'w'))
sys.argv = ['blender', '--', '--jobs', os.path.join(tmp, 'jobs.json'), '--out', tmp, '--device', 'GPU', '--samples', '32']
sys.path.insert(0, os.path.join(root, 'blender'))
t = time.time()
import render_faces  # noqa: E402
render_faces.main()
print('SMOKE TEST two full-resolution faces: %.1f s total -> %s' % (time.time() - t, tmp))
