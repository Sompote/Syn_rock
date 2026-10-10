"""Generate and label synthetic tunnel faces with Blender (Cycles), for Linux GPU boxes and local runs.

Jobs are drawn exactly as in batch_tunnel_faces.py (per-class or mixed, with the
optional calibration), split across --workers parallel Blender processes, then
labelled in a process pool and merged once into OUT/dataset/_annotations.coco.json.
Seeds that already have a labelled image are skipped, so an interrupted run resumes.

  python3 scripts/blender_batch.py --mixed 1000 --first-seed 10000 --out /data/faces --workers 3
"""
import argparse, json, os, shutil, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import batch_tunnel_faces as plan  # noqa: E402
import label_tunnel_face as lab    # noqa: E402
sys.path.insert(0, str(ROOT))
from tunnelgen.sampler import sample_job  # noqa: E402


def find_blender(given):
    for c in [given, os.environ.get('BLENDER'), shutil.which('blender'), '/Applications/Blender.app/Contents/MacOS/Blender']:
        if c and Path(c).exists():
            return c
    raise SystemExit('Blender not found: pass --blender or set BLENDER')


def label_job(args):
    job, out = args
    try:
        return job['seed'], lab.label_one(job, out), None
    except Exception as e:  # keep the batch going; report at the end
        return job['seed'], None, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=0)
    ap.add_argument('--codes', default=','.join(plan.CODES))
    ap.add_argument('--mixed', type=int, default=0)
    ap.add_argument('--foundation', type=int, default=0, help='this many faces from tunnelgen.sampler (GSI + RMC labels)')
    ap.add_argument('--first-seed', type=int, default=101)
    ap.add_argument('--out', default=str(ROOT / 'output'))
    ap.add_argument('--calibration', default=str(ROOT / 'calibration/class_calibration.json'))
    ap.add_argument('--blender', default=None)
    ap.add_argument('--workers', type=int, default=1, help='parallel Blender processes (2-3 keep one GPU busy)')
    ap.add_argument('--device', default='GPU', choices=['GPU', 'CPU'])
    ap.add_argument('--samples', type=int, default=32)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--prune', action='store_true', help='delete the large RGB renders once labelled (index masks are kept)')
    a = ap.parse_args()
    out = Path(a.out).resolve(); (out / 'render').mkdir(parents=True, exist_ok=True)
    if a.calibration and Path(a.calibration).exists():
        plan.CAL.update(json.loads(Path(a.calibration).read_text())); print('calibration:', a.calibration)

    if a.foundation:
        jobs = [sample_job(a.first_seed + k) for k in range(a.foundation)]
    elif a.mixed:
        jobs = [plan.mixed_job(a.first_seed + k) for k in range(a.mixed)]
    else:
        n = a.per_class or 1
        jobs = [plan.job_for(a.first_seed + i * n + k, 'RMC' + c) for i, c in enumerate(a.codes.split(',')) for k in range(n)]
    todo = [j for j in jobs if not (out / 'dataset/images' / ('tunnel_face_%03d.jpg' % j['seed'])).exists()]
    print('%d jobs, %d to render' % (len(jobs), len(todo)), flush=True)
    for j in todo:
        (out / 'render' / ('tunnel_face_%03d_job.json' % j['seed'])).write_text(json.dumps(j, indent=1))

    # render: one Blender process per chunk, all in parallel
    t0 = time.time(); blender = find_blender(a.blender); procs = []
    for w in range(min(a.workers, len(todo))):
        chunk = todo[w::a.workers]
        jf = out / ('jobs_worker%d.json' % w); jf.write_text(json.dumps(chunk))
        log = open(out / ('blender_worker%d.log' % w), 'w')
        procs.append((subprocess.Popen([blender, '-b', '--factory-startup', '-P', str(ROOT / 'blender/render_faces.py'), '--',
                                        '--jobs', str(jf), '--out', str(out), '--device', a.device,
                                        '--samples', str(a.samples), '--scale', str(a.scale)],
                                       stdout=log, stderr=subprocess.STDOUT), log, len(chunk)))
    for p, log, n in procs:
        p.wait(); log.close()
        if p.returncode:
            print('a Blender worker exited with code %d; see %s' % (p.returncode, log.name), flush=True)
    t1 = time.time()

    # label in parallel, merge once
    rendered = [j for j in todo if (out / 'render' / ('tunnel_face_%03d_rgb.png' % j['seed'])).exists()]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        results = list(ex.map(label_job, [(j, str(out)) for j in rendered]))
    ok = [res for _, res, err in results if err is None]
    lab.merge_coco(out, [(r['image'], r['rmc']) for r in ok])
    if any(r['gsi'] is not None for r in ok):
        lab.merge_coco(out, [(r['image'], r['gsi']) for r in ok if r['gsi'] is not None], 'gsi')
    for seed, res, err in results:
        if err:
            print('seed %d: labelling failed: %s' % (seed, err), flush=True)
        elif a.prune:
            (out / 'render' / ('tunnel_face_%03d_rgb.png' % seed)).unlink(missing_ok=True)
    t2 = time.time()
    print('rendered %d/%d in %.0f s (%.1f s/face wall), labelled %d in %.0f s' %
          (len(rendered), len(todo), t1 - t0, (t1 - t0) / max(1, len(rendered)), len(ok), t2 - t1), flush=True)


if __name__ == '__main__':
    main()
