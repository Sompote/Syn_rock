"""Tunnel-face scene geometry in NumPy, independent of any renderer.

A vectorised port of scripts/rhino_tunnel_face.py: the same hash, value noise,
fBm, joint sets, Voronoi blast facets, zones and boundaries, profile, lining,
floor, muck pile and fragments, with the same constants and evaluation order, so
one job file gives the same scene in Rhino and in Blender. build_scene(job)
returns plain arrays (vertices, faces, sRGB vertex colours, label class) plus the
camera and lights, which a renderer turns into images.
"""
import math, random
import numpy as np

W2, ZS, R = 5.0, 4.0, 5.0                      # half width, springline height, arch radius (m)
LEN = 2 * ZS + math.pi * R                     # length of the excavation line
STAIN_RGB = np.array([156., 94., 50.])
M32 = 0xffffffff


# ---------- noise (bit-identical to the Rhino version) ----------
def h2(i, j, s):
    i = np.asarray(i, np.int64); j = np.asarray(j, np.int64)
    n = (i * 374761393 + j * 668265263 + int(s) * 982451653) & M32
    n = ((n ^ (n >> 13)) * 1274126177) & M32
    return ((n ^ (n >> 16)) & 0xffff) / 65535.0


def vnoise(x, y, s):
    x = np.asarray(x, float); y = np.asarray(y, float)
    i = np.floor(x); j = np.floor(y); fx = x - i; fy = y - j
    i = i.astype(np.int64); j = j.astype(np.int64)
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy)
    a = h2(i, j, s); b = h2(i + 1, j, s); c = h2(i, j + 1, s); d = h2(i + 1, j + 1, s)
    return (a + (b - a) * fx) + ((c + (d - c) * fx) - (a + (b - a) * fx)) * fy - .5


def fbm(x, y, s, octaves=4):
    v = 0.; amp = 1.; f = 1.
    for k in range(octaves):
        v = v + amp * vnoise(np.asarray(x) * f, np.asarray(y) * f, s + 17 * k); amp *= .5; f *= 2.03
    return v


def voronoi(x, z, cell, s):
    """Nearest jittered-grid feature point: cell ids, offset to it, and gap to the second nearest (m)."""
    gx, gz = x / cell, z / cell
    i0 = np.floor(gx).astype(np.int64); j0 = np.floor(gz).astype(np.int64)
    best = np.full(gx.shape, 9.); second = np.full(gx.shape, 9.)
    bi = np.zeros(gx.shape, np.int64); bj = np.zeros(gx.shape, np.int64)
    box = np.zeros(gx.shape); boz = np.zeros(gx.shape)
    for di in (-1, 0, 1):                      # same visiting order as the scalar version
        for dj in (-1, 0, 1):
            i = i0 + di; j = j0 + dj
            fx = i + h2(i, j, s); fz = j + h2(i, j, s + 1)
            d2 = (gx - fx) ** 2 + (gz - fz) ** 2
            better = d2 < best
            mid = (~better) & (d2 < second)
            second = np.where(better, best, np.where(mid, d2, second))
            best = np.where(better, d2, best)
            bi = np.where(better, i, bi); bj = np.where(better, j, bj)
            box = np.where(better, (gx - fx) * cell, box); boz = np.where(better, (gz - fz) * cell, boz)
    return bi, bj, box, boz, (np.sqrt(second) - np.sqrt(best)) * cell


# ---------- scene description ----------
def make_sets(joint_sets, zr):
    out = []
    for k, (ang, sp) in enumerate(joint_sets):     # trace angle in face plane (deg), spacing (m)
        a = math.radians(ang + zr.uniform(-6, 6))
        out.append({'nx': math.cos(a), 'nz': math.sin(a), 'sp': sp * zr.uniform(.85, 1.15),
                    'off': zr.uniform(0, sp), 'seed': 100 + 31 * k})
    return out


def zones_of_job(job):
    seed = job['seed']; base = job.get('rock_rgb', [112, 106, 98])
    zone_jobs = job.get('zones') or [{
        'code': job.get('rmc_code'), 'rock_rgb': base, 'joint_sets': job['joint_sets'],
        'block_relief_m': job.get('block_relief_m', .16), 'facet_scale': job.get('facet_scale', 1.), 'crush': job.get('crush', 0.),
        'weathering': job.get('weathering', 0.), 'stained_joint_fraction': job.get('stained_joint_fraction', .3),
        'joint_dark': job.get('joint_dark', 1.), 'facet_dark': job.get('facet_dark', .45), 'wet_height_m': job.get('wet_height_m', 1.6),
        'grain': job.get('grain', .10), 'weathered_rgb': job.get('weathered_rgb', [150, 108, 70])}]
    return [{'sets': make_sets(zj['joint_sets'], random.Random(seed * 13 + zi)), 'rgb': np.array(zj['rock_rgb'], float),
             'relief': zj['block_relief_m'], 'facet': zj['facet_scale'], 'crush': zj['crush'], 'wz': zj['weathering'],
             'stain': zj['stained_joint_fraction'], 'dark': zj['joint_dark'], 'fdark': zj['facet_dark'],
             'wet': zj['wet_height_m'], 'code': zj['code'], 'grain': zj.get('grain', .10),
             'wcol': np.array(zj.get('weathered_rgb', [150, 108, 70]), float),
             'gsi': zj.get('gsi'), 'bedding': zj.get('bedding'), 'veins': zj.get('veins', []), 'karst': zj.get('karst', []),
             'grain_scale': zj.get('grain_scale', 0.), 'bump': zj.get('bump', 0.)} for zi, zj in enumerate(zone_jobs)]


class Scene:
    def __init__(self, job):
        self.job = job; self.seed = job['seed']
        self.zones = zones_of_job(job); self.bounds = job.get('boundaries', [])

    def zone_of(self, x, z):
        """Zone index and distance (m) to the nearest boundary trace."""
        seed = self.seed
        zi = np.zeros(np.shape(x), np.int64); near = np.full(np.shape(x), 99.)
        for bi, b in enumerate(self.bounds):
            a = math.radians(b['angle_deg']); nx_, nz_ = math.cos(a), math.sin(a)
            bs = seed + 600 + 37 * bi
            wp = b.get('warp_m', 0.)
            wx = x + wp * fbm(x * .3, z * .3, bs + 1); wz = z + wp * fbm(x * .3, z * .3, bs + 2)
            along = -nz_ * wx + nx_ * (wz - 4.3)
            wav = b.get('wav_m', .3)
            sd = nx_ * wx + nz_ * (wz - 4.3) - b['offset_m'] + wav * fbm(along * .22, 3, bs) + .4 * wav * fbm(along * .8, 5, bs + 3)
            rough = b.get('rough_m', 0.)
            if rough:
                sd = sd + rough * fbm(along * 2.5, 7, bs + 4, 3)
            jag = b.get('jag_m', 0.)
            if jag:
                u = along / b.get('jag_len_m', .7); k = np.floor(u); f = u - k
                sd = sd + jag * ((f - .5) * b.get('jag_ramp', .6) + (h2(k.astype(np.int64), bi, bs) - .5))
            zi = zi + (sd > 0)
            near = np.minimum(near, np.abs(sd))
        return np.minimum(zi, len(self.zones) - 1), near

    def joint_field(self, x, z, P):
        """Relief toward the camera, joint darkening, iron staining and block tone at face points (x, z)."""
        seed = self.seed
        keys = []; d = np.zeros_like(x); dark = np.zeros_like(x); stain = np.zeros_like(x)
        for k, js in enumerate(P['sets']):
            along = -js['nz'] * x + js['nx'] * z
            t = (js['nx'] * x + js['nz'] * z) + .12 * fbm(along * .35, k, js['seed']) + js['off']
            q = t / js['sp']; idx0 = np.floor(q).astype(np.int64)
            lo0 = idx0 + .35 * (h2(idx0, k, seed + 5) - .5); hi0 = idx0 + 1 + .35 * (h2(idx0 + 1, k, seed + 5) - .5)
            m1 = q < lo0; m2 = (~m1) & (q >= hi0)       # irregular spacing: joint shifted past q
            idx = np.where(m1, idx0 - 1, np.where(m2, idx0 + 1, idx0))
            lo = np.where(m1, idx + .35 * (h2(idx, k, seed + 5) - .5), np.where(m2, hi0, lo0))
            hi = np.where(m1, lo0, np.where(m2, idx + 1 + .35 * (h2(idx + 1, k, seed + 5) - .5), hi0))
            u = (q - lo) / (hi - lo); dist = js['sp'] * np.minimum(q - lo, hi - q)
            jn = np.where(q - lo < hi - q, idx, idx + 1)
            persist = np.clip(.5 + 2.2 * fbm(along * .45, jn * 1.7, js['seed'] + 9), 0., 1.)
            keys.append(idx)
            dark = np.maximum(dark, persist * np.exp(-(dist / (.012 + .02 * h2(jn, k, seed + 6))) ** 2))
            stained = h2(jn, k, seed + 7) < P['stain']
            val = persist * np.exp(-(dist / .16) ** 2) * (.5 + .6 * fbm(x * 2, z * 2, 55))
            stain = np.where(stained, np.maximum(stain, val), stain)
            d = d + (h2(idx, k, seed) - .5) * .05 * persist + (h2(idx, k, seed + 3) - .5) * .5 * (u - .5) * js['sp'] * .4
        bk = h2(keys[0] * 7 + keys[1], keys[2] if len(keys) > 2 else 0, seed + 11)
        d = d + (bk - .5) * P['relief']
        fs = P['facet']; cr = P['crush']
        scales = [(.55, .07 * fs, .55 * fs, 300, P['fdark']), (.22, .025 * fs, .45 * fs, 400, 0.)]
        if cr > 0:
            scales.append((.09, .02 * cr, .5 * cr, 500, .5 * cr))
        for cell, amp, tilt, s2, gd in scales:
            ci, cj, ox, oz, gap = voronoi(x, z, cell, seed + s2)
            d = d + (h2(ci, cj, s2) - .5) * amp + ((h2(ci, cj, s2 + 2) - .5) * ox + (h2(ci, cj, s2 + 3) - .5) * oz) * tilt
            if gd > 0:
                dark = np.maximum(dark, gd * np.exp(-(gap / .012) ** 2))
        return d, dark * P['dark'], stain, bk

    def patterns(self, x, z, P):
        """Bedding/lamination (optionally folded), veins and karst: relief, darkening, colour factor and vein blend."""
        seed = self.seed; d = np.zeros_like(x); dark = np.zeros_like(x); tone = np.ones_like(x)
        vein = np.zeros_like(x); vein_rgb = np.zeros((len(x), 3))
        b = P['bedding']
        if b:
            a = math.radians(b['angle_deg']); nx_, nz_ = math.cos(a), math.sin(a)
            along = -nz_ * x + nx_ * z
            t = (nx_ * x + nz_ * z + b['fold_amp_m'] * np.sin(2 * math.pi * along / b['fold_len_m'])
                 + .04 * fbm(along * .3, 1, seed + 800))
            q = t / b['spacing_m']; layer = np.floor(q).astype(np.int64); f = q - layer
            dist = b['spacing_m'] * np.minimum(f, 1 - f)
            tone = 1 + b['band_contrast'] * 2 * (h2(layer, 7, seed + 801) - .5)
            d = d + (h2(layer, 8, seed + 802) - .5) * b['relief_m']
            dark = np.maximum(dark, b['dark'] * np.exp(-(dist / .012) ** 2) * (h2(layer, 9, seed + 803) > .3))
        for k, v in enumerate(P['veins']):
            a = math.radians(v['angle_deg']); nx_, nz_ = math.cos(a), math.sin(a)
            along = -nz_ * x + nx_ * z
            sd = nx_ * x + nz_ * (z - 4.3) - v['offset_m'] + .12 * fbm(along * .5, k, seed + 820 + k)
            w = np.exp(-(sd / v['width_m']) ** 2) * np.clip(.6 + 1.5 * fbm(along * .4, k + 9, seed + 840), 0, 1)
            new = w > vein
            vein_rgb[new] = np.array(v['rgb'], float); vein = np.maximum(vein, w)
            d = d + .008 * w
        for c in P['karst']:
            rr = np.hypot(x - c['x'], z - c['z']) / c['r'] * (1 + .25 * fbm(x * 2, z * 2, seed + 860))
            m = np.clip(1 - rr ** 2, 0, 1)
            d = d - c['depth'] * m ** .7
            dark = np.maximum(dark, .85 * m ** .5)
        return d, dark, tone, vein, vein_rgb

    def seepage(self, x, z):
        """Dark wet streaks running down the face from joints, one per seepage point."""
        rs = np.random.default_rng(self.seed + 77); out = np.zeros_like(x)
        for k in range(self.job.get('seepage', 0)):
            x0, z0 = rs.uniform(-4.5, 4.5), rs.uniform(2., 8.5); w, L = rs.uniform(.05, .25), rs.uniform(1., 5.)
            wx = x - x0 - .25 * fbm(z * .8, k, self.seed + 900)
            out = np.maximum(out, np.exp(-(wx / w) ** 2) * (z < z0) * np.exp(-np.maximum(0, z0 - z) / L))
        return out

    def rock_colour(self, x, z, dark, stain, bk, wet, P):
        seed = self.seed
        tone = 1 + (bk - .5) * .22 + .18 * fbm(x * 1.3, z * 1.3, seed + 40) \
            + P['grain'] * (h2(np.trunc(x * 60).astype(np.int64), np.trunc(z * 60).astype(np.int64), seed) - .5)
        c = P['rgb'][None, :] * tone[:, None]
        if P['wz'] > 0:
            w = np.clip(P['wz'] * (.75 + .6 * fbm(x * .5, z * .5, seed + 45)), 0., 1.)[:, None]
            wc = P['wcol'][None, :] * (1 + .15 * fbm(x * 2.5, z * 2.5, seed + 46))[:, None]
            c = c * (1 - w) + wc * w
        s = stain[:, None]
        c = c * (1 - s * .55) + STAIN_RGB[None, :] * s * .55
        c = c * (1 - .48 * dark)[:, None]
        return c * (1 - np.array([.28, .26, .2])[None, :] * wet[:, None])


def profile(s):
    """Points and outward normals on the excavation line, s from the right foot to the left foot."""
    s = np.asarray(s, float)
    a = (s - ZS) / R
    px = np.where(s <= ZS, W2, np.where(s >= ZS + math.pi * R, -W2, R * np.cos(a)))
    pz = np.where(s <= ZS, s, np.where(s >= ZS + math.pi * R, LEN - s, ZS + R * np.sin(a)))
    qx = np.where(s <= ZS, 1., np.where(s >= ZS + math.pi * R, -1., np.cos(a)))
    qz = np.where(s <= ZS, 0., np.where(s >= ZS + math.pi * R, 0., np.sin(a)))
    return px, pz, qx, qz


def grid_faces(nu, nv):
    j, i = np.mgrid[0:nv - 1, 0:nu - 1]
    a = (j * nu + i).ravel()
    return np.stack([a, a + 1, a + nu + 1, a + nu], 1)


def to_u8(c):
    return np.clip(c, 0, 255).astype(np.uint8)        # Rhino clamp() truncates toward zero too


def mesh(name, cls, verts, faces, colours, flat=False, wet=None, zone=None):
    v = np.asarray(verts, np.float32)
    return {'name': name, 'cls': cls, 'verts': v, 'faces': np.asarray(faces, np.int32), 'colours': to_u8(colours), 'flat': flat,
            'wet': np.zeros(len(v), np.float32) if wet is None else np.asarray(wet, np.float32), 'zone': zone}


def split_by_zone(name, cls_prefix, verts, cols, zone, nu, nv, wet=None):
    faces = grid_faces(nu, nv); fz = zone[faces[:, 0]]
    return [mesh('%s_zone%d' % (name, z), '%s%d' % (cls_prefix, z), verts, faces[fz == z], cols, wet=wet, zone=z)
            for z in sorted(set(fz.tolist()))]


def box_mesh(name, cls, centre, size, axes, rgb, segments=1):
    """Box with `segments` colour bands along its first axis (bolt plates, steel-set pieces, the scale bar)."""
    c = np.asarray(centre, float); A = [np.asarray(a, float) / np.linalg.norm(a) for a in axes]
    vs, fs, cs = [], [], []
    for k in range(segments):
        lo = -.5 + k / segments; hi = lo + 1. / segments
        corners = [c + A[0] * size[0] * a + A[1] * size[1] * b + A[2] * size[2] * e
                   for a in (lo, hi) for b in (-.5, .5) for e in (-.5, .5)]
        base = len(vs); vs += corners
        fs += [[base + i for i in q] for q in ([0, 1, 3, 2], [4, 6, 7, 5], [0, 2, 6, 4], [1, 5, 7, 3], [0, 4, 5, 1], [2, 3, 7, 6])]
        col = rgb[k % len(rgb)] if isinstance(rgb[0], (list, tuple)) else rgb
        cs += [col] * 8
    return mesh(name, cls, vs, fs, np.array(cs, float), flat=True)


def merge(ms, name, cls):
    v, f, c, n = [], [], [], 0
    for m in ms:
        v.append(m['verts']); f.append(m['faces'] + n); c.append(m['colours']); n += len(m['verts'])
    return mesh(name, cls, np.vstack(v), np.vstack(f), np.vstack(c).astype(float), flat=True)


def icosphere1():
    """Unit icosphere with one subdivision (42 vertices, 80 triangles), like Rhino's CreateIcoSphere(.., 1)."""
    t = (1 + 5 ** .5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
         (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = [np.array(p, float) / np.linalg.norm(p) for p in v]; cache = {}; nf = []

    def mid(a, b):
        key = (min(a, b), max(a, b))
        if key not in cache:
            m = v[a] + v[b]; v.append(m / np.linalg.norm(m)); cache[key] = len(v) - 1
        return cache[key]
    for a, b, c in f:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    return np.array(v), np.array(nf)


def build_scene(job):
    S = Scene(job); seed = S.seed; rng = random.Random(seed)
    base = np.array(job.get('rock_rgb', [112, 106, 98]), float)
    meshes = []

    # 1. rock face: height field, split into one mesh per zone
    step = job.get('face_grid_m', .035)
    nu = int(2 * (W2 + .45) / step) + 1; nv = int((ZS + R + .45) / step) + 1
    jj, ii = np.mgrid[0:nv, 0:nu]
    x = (-W2 - .45 + ii * step).ravel().astype(float); z = (jj * step).ravel().astype(float)
    zone, near = S.zone_of(x, z)
    gw = max([b.get('gouge_m', 0.) for b in S.bounds] + [0.])
    gouge = np.exp(-(near / gw) ** 2) if gw > 0 else np.zeros_like(x)
    d = np.zeros_like(x); cols = np.zeros((len(x), 3)); wetv = np.zeros_like(x)
    for zi, P in enumerate(S.zones):
        m = zone == zi
        if not m.any():
            continue
        xm, zm, g = x[m], z[m], gouge[m]
        dd, dark, stain, bk = S.joint_field(xm, zm, P)
        pd, pdark, ptone, vein, vein_rgb = S.patterns(xm, zm, P)
        dd = dd + pd; dark = np.maximum(dark, pdark)
        dd = dd * (1 - g) - .12 * g
        r2 = (xm / W2) ** 2 * .55 + ((zm - 4.3) / 4.6) ** 2 * .45
        dd = dd + .32 * (1 - r2) + .05 * fbm(xm * 1.6, zm * 1.6, seed + 20) + .012 * fbm(xm * 9, zm * 9, seed + 21, 2)
        we = P['wet']
        wet = np.clip((we - zm) / we + .9 * fbm(xm * .45, zm * .45, seed + 90) + .8 * g, 0., 1.)
        if 'wet_level' in job:                          # groundwater condition scales the wet patches, plus seepage streaks
            wet = np.clip(wet * (.3 + job['wet_level']) + S.seepage(xm, zm), 0., 1.)
        c = S.rock_colour(xm, zm, dark, stain, bk, wet, P) * ptone[:, None]
        c = c * (1 - .8 * vein)[:, None] + vein_rgb * (.8 * vein)[:, None]
        c = c * (1 - .55 * g)[:, None] + np.array([70., 60., 50.])[None, :] * (.55 * g)[:, None]
        d[m] = dd; cols[m] = c; wetv[m] = wet
    verts = np.stack([x, -d, z], 1)
    x_face, z_face, d_face = x, z, d
    meshes += split_by_zone('rock_face', 'rock', verts, cols, zone, nu, nv, wetv)

    # 2. unsupported rock strip between the face and the shotcrete
    ns = int(LEN / .04) + 1; ny = int(2.2 / .04) + 1
    jj, ii = np.mgrid[0:ny, 0:ns]
    s = (ii * LEN / (ns - 1)).ravel(); y = (-1.9 + jj * .04).ravel()
    px, pz, qx, qz = profile(s)
    r = .14 + .07 * fbm(s * 1.2, y * 1.2, seed + 60) + .02 * fbm(s * 7, y * 7, seed + 61, 2)
    barrel = np.zeros_like(s)
    if job.get('half_barrels'):                         # half-casts of perimeter blast holes, parallel to the tunnel axis
        sp = job.get('barrel_spacing_m', .6)
        ph = (s + .05 * fbm(y * .5, 3, seed + 870)) / sp
        barrel = ((np.maximum(0, np.cos(2 * math.pi * ph)) ** 6) * (h2(np.floor(ph + .5).astype(np.int64), 1, seed + 871) > .35)
                  * (y > -1.75))
        r = r - .035 * barrel
    wet = np.maximum(0., (1.2 - pz) / 1.2)
    zone, _ = S.zone_of(px, pz); cols = np.zeros((len(s), 3))
    for zi, P in enumerate(S.zones):
        m = zone == zi
        if m.any():
            _, dark, stain, bk = S.joint_field(px[m] + .3 * y[m], pz[m], P)
            cols[m] = S.rock_colour(px[m], pz[m] + y[m], dark * .25, stain * .3, bk, wet[m], P)
    cols = cols * (1 - .25 * barrel)[:, None]
    verts = np.stack([px + qx * r, y, pz + qz * r], 1)
    meshes += split_by_zone('rock_perimeter', 'rock', verts, cols, zone, ns, ny, wet * .6)

    # 3. shotcrete lining with a ragged front edge, and its front lip
    sc = np.array(job.get('shotcrete_rgb', [176, 170, 158]), float)
    front = lambda s: -1.45 - .35 * (fbm(s * .9, 0, seed + 70) + .5) - .12 * fbm(s * 5, 1, seed + 71, 2)
    nsl = int(LEN / .05) + 1; nyl = 120
    jj, ii = np.mgrid[0:nyl, 0:nsl]
    s = (ii * LEN / (nsl - 1)).ravel(); f = front(s); y = -16 + (f + 16) * jj.ravel() / (nyl - 1.)
    px, pz, qx, qz = profile(s)
    r = .015 * fbm(s * 4, y * 4, seed + 80) + .006 * fbm(s * 15, y * 15, seed + 81, 2)
    tone = 1 + .12 * fbm(s * .8, y * .8, seed + 82) + .05 * (h2(np.trunc(s * 50).astype(np.int64), np.trunc(y * 50).astype(np.int64), seed + 83) - .5)
    wet = np.maximum(0., (1. - pz) / 1. + .6 * fbm(s * .5, y * .3, seed + 84))
    cols = sc[None, :] * (tone * (1 - .35 * np.minimum(1, wet)))[:, None]
    meshes.append(mesh('shotcrete_lining', 'shotcrete', np.stack([px + qx * r, y, pz + qz * r], 1), grid_faces(nsl, nyl), cols,
                       wet=np.clip(wet, 0, 1) * .5))
    jj, ii = np.mgrid[0:4, 0:nsl]
    s = (ii * LEN / (nsl - 1)).ravel(); jv = jj.ravel().astype(float); f = front(s)
    px, pz, qx, qz = profile(s); r = .24 * jv / 3.
    tone = .82 + .1 * fbm(s * 3, jv, seed + 85)
    meshes.append(mesh('shotcrete_edge', 'shotcrete', np.stack([px + qx * r, f + .02 * jv, pz + qz * r], 1),
                       grid_faces(nsl, 4), sc[None, :] * tone[:, None]))

    # 4. floor, muck pile and fragments
    mud = np.array(job.get('floor_rgb', [92, 80, 66]), float)
    jj, ii = np.mgrid[0:165, 0:101]
    x = (-W2 + ii * .1).ravel(); y = (-16 + jj * .1).ravel()
    t = 1 + .25 * fbm(x * .6, y * .6, seed + 95) + .08 * fbm(x * 6, y * 6, seed + 96, 2)
    meshes.append(mesh('floor', 'other', np.stack([x, y, .01 * fbm(x * 3, y * 3, seed + 97)], 1), grid_faces(101, 165), mud[None, :] * t[:, None]))

    def muck_h(x, y):
        b = 1 - (x / 5.8) ** 2 - ((y - .2) / 4.0) ** 2
        return np.maximum(0., b) ** .8 * job.get('muck_height_m', 1.5) + .12 * fbm(x * 1.4, y * 1.4, seed + 100) * np.maximum(0., b)
    jj, ii = np.mgrid[0:93, 0:201]
    x = (-W2 + ii * .05).ravel(); y = (-4.4 + jj * .05).ravel()
    t = .75 + .3 * fbm(x * 3, y * 3, seed + 101) + .15 * (h2(ii.ravel(), jj.ravel(), seed + 102) - .5)
    meshes.append(mesh('muck_pile', 'other', np.stack([x, y, muck_h(x, y)], 1), grid_faces(201, 93),
                       base[None, :] * t[:, None] * np.array([.9, .88, .86])[None, :]))

    ico_v, ico_f = icosphere1(); fv, ff, fc = [], [], []
    for k in range(job.get('fragments', 700)):
        x = rng.uniform(-4.6, 4.6); y = rng.uniform(-3.8, 0.)
        if float(muck_h(np.array(x), np.array(y))) < .05:
            continue
        a = rng.uniform(.05, .2) * (2.2 if rng.random() < .1 else 1)
        sx, sy, sz = rng.uniform(.8, 1.3), rng.uniform(.6, 1.1), rng.uniform(.35, .7)
        v = ico_v * a * np.array([[rng.uniform(.7, 1.15)] for _ in range(len(ico_v))]) * np.array([sx, sy, sz])
        az = rng.uniform(0, 6.3); ax = rng.uniform(-.5, .5)
        rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
        rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
        v = v @ rz.T @ rx.T + np.array([x, y, float(muck_h(np.array(x), np.array(y))) + a * .2])
        t = rng.uniform(.42, .85)
        ff.append(ico_f + sum(len(p) for p in fv)); fv.append(v); fc.append(np.tile(base * t, (len(v), 1)))
    if fv:
        meshes.append(mesh('muck_fragments', 'other', np.vstack(fv), np.vstack(ff), np.vstack(fc), flat=True))

    # 5. support and occluders
    rs = np.random.default_rng(seed + 31)
    if job.get('bolts'):                                 # rock-bolt plates on the lining
        plates = []
        for sb in np.arange(rs.uniform(.6, 1.2), LEN - .6, rs.uniform(1.2, 1.7)):
            px_, pz_, qx_, qz_ = [float(v[0]) for v in profile(np.array([sb]))]
            if pz_ < 1.:
                continue
            n = np.array([qx_, 0., qz_]); t = np.array([-qz_, 0., qx_])
            for yb in np.arange(-2.6 - rs.uniform(0, .5), -15, -rs.uniform(1.3, 1.8)):
                grey = float(rs.integers(60, 110))
                plates.append(box_mesh('bolt', 'shotcrete', np.array([px_, yb, pz_]) - n * .03, (.16, .16, .03),
                                       (t, [0, 1, 0], n), [grey, grey, grey]))
        if plates:
            meshes.append(merge(plates, 'bolt_plates', 'shotcrete'))
    if job.get('steel_sets'):                            # steel arches, partly sprayed over
        ribs = []; ss = np.linspace(0, LEN, 160); px_, pz_, qx_, qz_ = profile(ss)
        for yb in np.arange(-2.2 - rs.uniform(0, .6), -15, -rs.uniform(1., 1.5)):
            grey = float(rs.integers(55, 95))
            for k in range(len(ss) - 1):
                n = np.array([qx_[k], 0., qz_[k]]); t = np.array([px_[k + 1] - px_[k], 0., pz_[k + 1] - pz_[k]])
                ribs.append(box_mesh('rib', 'shotcrete', np.array([px_[k], yb, pz_[k]]) + t / 2 - n * .06,
                                     (np.linalg.norm(t), .16, .08), (t, [0, 1, 0], n), [grey, grey, grey]))
        meshes.append(merge(ribs, 'steel_sets', 'shotcrete'))
    if job.get('scale_bar'):                             # 0.5 m black-and-white scale bar held against the face
        xs, zs = rs.uniform(-2.5, 2.5), rs.uniform(1.6, 3.5)
        near = int(np.argmin((x_face - xs) ** 2 + (z_face - zs) ** 2))
        meshes.append(box_mesh('scale_bar', 'other', [xs, -d_face[near] - .05, zs], (.5, .02, .07),
                               ([1, 0, rs.uniform(-.2, .2)], [0, 1, 0], [0, 0, 1]), [(235, 235, 230), (20, 20, 20)], segments=10))
    if job.get('duct'):                                  # ventilation duct hung at the crown corner
        side = float(rs.choice([-1., 1.])); rad = rs.uniform(.35, .55); cx0, cz0 = side * rs.uniform(2.4, 3.4), rs.uniform(6.6, 7.8)
        th = np.linspace(0, 2 * math.pi, 33)[:-1]; yy = np.linspace(-16, -rs.uniform(2.5, 5.), 30)
        T, Y = np.meshgrid(th, yy)
        dv = np.stack([cx0 + rad * np.cos(T).ravel(), Y.ravel(), cz0 + rad * np.sin(T).ravel()], 1)
        j, i = np.mgrid[0:len(yy) - 1, 0:len(th)]
        a0 = (j * len(th) + i).ravel(); a1 = (j * len(th) + (i + 1) % len(th)).ravel()
        tone = [(200, 180, 60), (150, 150, 150), (60, 60, 60)][int(rs.integers(0, 3))]
        meshes.append(mesh('duct', 'other', dv, np.stack([a0, a1, a1 + len(th), a0 + len(th)], 1),
                           np.tile(np.array(tone, float), (len(dv), 1)) * (1 + .1 * np.cos(T).ravel())[:, None]))

    cam = job['camera']
    lights = [{'name': 'key', 'type': 'POINT' if job.get('flash') else 'SPOT', 'location': cam['key_light'],
               'target': cam.get('target', [0, 0, 4.2]) if job.get('flash') else [0, 0, 4.2],
               'intensity': job.get('key_intensity', .7), 'spot_deg': 48},
              {'name': 'fill', 'type': 'POINT', 'location': [-3.2, -5.5, 1.6], 'target': [0, 0, 3],
               'intensity': job.get('fill_intensity', .12)}]
    return {'meshes': meshes, 'camera': cam, 'lights': lights, 'zones': [z['code'] for z in S.zones],
            'spot_blend': job.get('spot_blend', .45), 'shadow_soft': job.get('shadow_soft', .15),
            'zone_gsi': [z['gsi'] for z in S.zones],
            'zone_params': [{'grain': z['grain'], 'grain_scale': z['grain_scale'], 'bump': z['bump']} for z in S.zones],
            'ambient_rgb': job.get('ambient_rgb', [16, 15, 14]), 'light_kelvin': job.get('light_kelvin'),
            'exposure': job.get('exposure', 0.)}
