"""Render and label Rhino tunnel faces for each RMC class, one randomised job per seed.

Each seed draws its own rock colour, joint sets, relief, muck pile, lining tone
and camera; the job is kept in render/<name>_job.json so a face can be rebuilt
or re-labelled with another RMC code later.
"""
import argparse, json, random, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import rhino_bridge
ROCKS = {'granite_grey': [112, 106, 98], 'sandstone_tan': [138, 118, 92], 'dark_shale': [74, 72, 70],
         'greenstone': [88, 98, 86], 'red_siltstone': [120, 88, 72], 'limestone_light': [150, 146, 136]}


# Assumed severity order A (best) -> E (worst), 01 -> 03 worse within a letter.
# The source dataset has no RMC codebook; these are visual archetypes, not rated rock masses.
# Each entry: (best, worst) for joint sets, spacing m, block relief m, facet scale, crush,
# weathering, stained fraction, facet-outline darkness (fdark), joint darkness, wet height m, muck height m.
CLASSES = {
    'A': dict(fdark=(.08, .15), sets=(1, 2), sp=(2.6, 1.8), relief=(.03, .05), facet=(.5, .6), crush=(0, 0), wz=(0, .02), stain=(0, .1), dark=(.6, .7), wet=(.6, 1.), muck=(.6, 1.)),
    'B': dict(fdark=(.18, .28), sets=(2, 3), sp=(1.6, .95), relief=(.06, .09), facet=(.7, .85), crush=(0, 0), wz=(0, .08), stain=(.1, .25), dark=(.7, .85), wet=(.8, 1.4), muck=(.8, 1.3)),
    'C': dict(fdark=(.35, .45), sets=(3, 3), sp=(.85, .45), relief=(.1, .15), facet=(.95, 1.05), crush=(0, .2), wz=(.08, .2), stain=(.2, .4), dark=(.85, 1.), wet=(1.2, 2.), muck=(1., 1.6)),
    'D': dict(fdark=(.45, .5), sets=(3, 4), sp=(.4, .2), relief=(.14, .2), facet=(1.05, 1.2), crush=(.35, .8), wz=(.2, .45), stain=(.35, .6), dark=(.9, 1.), wet=(1.8, 3.), muck=(1.3, 1.9)),
    'E': dict(fdark=(.25, .2), sets=(2, 3), sp=(.3, .15), relief=(.18, .26), facet=(.55, .4), crush=(.8, 1.), wz=(.6, .95), stain=(.3, .5), dark=(.45, .35), wet=(2.5, 4.), muck=(1.6, 2.2)),
}


def job_for(seed, code):
    r = random.Random(seed * 7919)
    rock = r.choice(sorted(ROCKS))
    tint = [max(30, min(200, v + r.randint(-10, 10))) for v in ROCKS[rock]]
    dist = r.uniform(8.5, 11.)
    cx = r.uniform(-1.5, 1.5)
    shot = r.randint(130, 175)
    job = {'seed': seed, 'rmc_code': code, 'rock_type': rock, 'rock_rgb': tint,
           'shotcrete_rgb': [shot, shot - 5, shot - 15], 'floor_rgb': [r.randint(70, 100), r.randint(62, 85), r.randint(50, 70)],
           'face_grid_m': .035, 'fragments': r.randint(350, 900),
           'key_intensity': r.uniform(.55, .85), 'fill_intensity': r.uniform(.05, .2),
           'camera': {'location': [cx, -dist, r.uniform(2.4, 3.6)], 'target': [cx * .3 + r.uniform(-.3, .3), 0, r.uniform(4.1, 4.7)],
                      'lens_mm': r.uniform(20, 26), 'key_light': [cx + r.uniform(-1.5, 1.5), -dist + 1.5, r.uniform(3.5, 5.)]},
           'capture_size': [1560, 1040]}
    z = zone_params(code, r, [r.uniform(0, 180) for _ in range(4)], tint)
    job.update({k: v for k, v in z.items() if k not in ('code', 'rock_rgb')})
    job['muck_height_m'] = z.pop('muck_height_m')
    return job


def zone_params(code, r, angles, rgb):
    """Rock-mass condition of one face zone for an RMC code; joint orientations are shared by the caller."""
    if code == 'Shortcrete00':
        raise SystemExit('Shortcrete00 is the lining class, not a face condition')
    c = CLASSES[code[3]]
    t = min(1., max(0., (int(code[4:]) - 1) / 2. + r.uniform(-.12, .12)))  # 01 -> 0, 02 -> .5, 03 -> 1
    lerp = lambda k: c[k][0] + (c[k][1] - c[k][0]) * t
    n = r.randint(*c['sets'])
    sp = lerp('sp')
    sets = [[angles[k], sp * r.uniform(.8, 1.25)] for k in range(n)]
    while len(sets) < 3:
        sets.append([angles[len(sets)], max(sp * 1.5, r.uniform(2.5, 4.))])  # sparse, widely spaced set
    return dict(code=code, rock_rgb=rgb, joint_sets=sets, block_relief_m=lerp('relief'), facet_scale=lerp('facet'),
                crush=lerp('crush'), weathering=lerp('wz'), stained_joint_fraction=lerp('stain'), joint_dark=lerp('dark'),
                facet_dark=lerp('fdark'), wet_height_m=lerp('wet'), muck_height_m=lerp('muck'), severity_t=t)


CODES = [L + s for L in 'ABCDE' for s in ('01', '02', '03')]


def irregular(r, scale):
    """Boundary shape: metre-scale bends, waviness, 0.2 m roughness and en-echelon steps (m)."""
    # fbm() has a typical amplitude of about +-0.3, so these are ~3x the intended lateral deviation
    return dict(warp_m=r.uniform(1., 2.4) * scale, wav_m=r.uniform(1.2, 2.4) * scale, rough_m=r.uniform(.12, .25),
                jag_m=r.uniform(.25, .6) * scale if r.random() < .6 else 0., jag_len_m=r.uniform(.4, 1.1),
                jag_ramp=r.uniform(.3, 1.))


def mixed_job(seed):
    """Face with 2 zones (contact) or 3 zones (a weaker fault band between two host zones)."""
    r = random.Random(seed * 104729)
    job = job_for(seed, 'RMCA01')
    for k in [k for k in job if k in ('rmc_code', 'joint_sets', 'block_relief_m', 'facet_scale', 'crush', 'weathering',
                                        'stained_joint_fraction', 'joint_dark', 'facet_dark', 'wet_height_m', 'severity_t')]:
        del job[k]
    angles = [r.uniform(0, 180) for _ in range(4)]
    host = r.randrange(0, 9)                                   # A01..C03
    if r.random() < .5:
        pattern = 'contact'
        other = min(14, host + r.randint(3, 8))
        codes = [CODES[host], CODES[other]] if r.random() < .5 else [CODES[other], CODES[host]]
        trace = r.uniform(35, 145)
        bounds = [dict(irregular(r, 1.), angle_deg=trace + 90, offset_m=r.uniform(-1.8, 1.8), gouge_m=0.)]
    else:
        pattern = 'fault_band'
        band = r.randrange(9, 15)                              # D01..E03
        side = min(8, max(0, host + r.randint(-2, 2)))
        codes = [CODES[host], CODES[band], CODES[side]]
        trace = r.uniform(40, 140); o1 = r.uniform(-2.5, .5)
        bounds = [dict(irregular(r, .7), angle_deg=trace + 90 + r.uniform(-10, 10), offset_m=o1 + w,
                       gouge_m=r.uniform(.05, .14)) for w in (0, r.uniform(1.4, 3.))]
    tint = job['rock_rgb']
    zones = [zone_params('RMC' + c, r, angles, [max(30, min(200, v + r.randint(-12, 12))) for v in tint]) for c in codes]
    for z in zones:
        z.pop('muck_height_m')
    job.update(rmc_code=None, pattern=pattern, zones=zones, boundaries=bounds,
               muck_height_m=CLASSES[max(codes)[0]]['muck'][0] + r.uniform(0, .4))
    return job


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=2)
    ap.add_argument('--first-seed', type=int, default=101)
    ap.add_argument('--codes', default=','.join(CODES))
    ap.add_argument('--mixed', type=int, default=0, help='instead of per-class faces, make this many 2-3 zone faces')
    ap.add_argument('--out', default=str(ROOT / 'output'))
    a = ap.parse_args()
    OUT = Path(a.out).resolve(); OUT.mkdir(parents=True, exist_ok=True)
    rhino_code = (ROOT / 'scripts/rhino_tunnel_face.py').read_text().replace('__OUT_DIR__', str(OUT))
    if a.mixed:
        plan = [(None, a.first_seed + k) for k in range(a.mixed)]
    else:
        plan = [('RMC' + c, a.first_seed + i * a.per_class + k) for i, c in enumerate(a.codes.split(',')) for k in range(a.per_class)]
    for code, seed in plan:
        job = mixed_job(seed) if code is None else job_for(seed, code)
        code = code or '+'.join(z['code'] for z in job['zones'])
        (OUT / 'job.json').write_text(json.dumps(job, indent=1))
        (OUT / 'render').mkdir(parents=True, exist_ok=True)
        (OUT / 'render' / ('tunnel_face_%03d_job.json' % seed)).write_text(json.dumps(job, indent=1))
        res = rhino_bridge.send('execute_rhinoscript_python_code', {'code': rhino_code}, timeout=600)
        if res.get('status') != 'success' or not res.get('result', {}).get('success', False):
            print('seed %d: Rhino failed\n%s' % (seed, json.dumps(res)[-2000:]), flush=True); continue
        r = subprocess.run([sys.executable, str(ROOT / 'scripts/label_tunnel_face.py'), '--out', str(OUT)], capture_output=True, text=True)
        if r.returncode != 0:
            print('seed %d: labelling failed\n%s' % (seed, r.stderr[-2000:]), flush=True); continue
        regions = json.loads(r.stdout)['regions']
        print('seed %d %s %s: %s' % (seed, code, job['rock_type'], {k: v['fraction'] for k, v in regions.items()}), flush=True)


if __name__ == '__main__':
    main()
