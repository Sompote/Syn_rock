"""Rhino 8 (IronPython 2.7): synthetic drill-and-blast tunnel face for RMC-style labelling.

Run through rhino_bridge.py. Builds a jointed rock face, the unsupported rock
strip, a shotcrete lining with a ragged front edge, a muck pile and the floor,
then captures an RGB view and a flat-colour ID view from the same camera.
Only objects tagged by this generator are ever deleted.
"""
import Rhino, System, math, random, json, os
from System.Drawing import Color
from System.Drawing.Imaging import ImageFormat

ROOT = '__OUT_DIR__'  # replaced with the absolute output folder by batch_tunnel_faces.py before sending
TAG = 'rock_v4_tunnel_face'
JOB = json.load(open(os.path.join(ROOT, 'job.json')))
doc = Rhino.RhinoDoc.ActiveDoc
if [o for o in doc.Objects if o.Attributes.GetUserString('generator') != TAG]:
    raise RuntimeError('Active document contains unrelated objects; open an empty document first.')
if doc.ModelUnitSystem != Rhino.UnitSystem.Meters:
    doc.AdjustModelUnitSystem(Rhino.UnitSystem.Meters, False)
view = doc.Views.ActiveView
vp = view.ActiveViewport

# ---------- noise ----------
def h2(i, j, s):
    n = (i * 374761393 + j * 668265263 + s * 982451653) & 0xffffffff
    n = ((n ^ (n >> 13)) * 1274126177) & 0xffffffff
    return ((n ^ (n >> 16)) & 0xffff) / 65535.0

def vnoise(x, y, s):
    i = int(math.floor(x)); j = int(math.floor(y)); fx = x - i; fy = y - j
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy)
    a = h2(i, j, s); b = h2(i + 1, j, s); c = h2(i, j + 1, s); d = h2(i + 1, j + 1, s)
    return (a + (b - a) * fx) + ((c + (d - c) * fx) - (a + (b - a) * fx)) * fy - .5

def fbm(x, y, s, octaves=4):
    v = 0.; amp = 1.; f = 1.
    for k in range(octaves):
        v += amp * vnoise(x * f, y * f, s + 17 * k); amp *= .5; f *= 2.03
    return v

def clamp(v): return int(max(0, min(255, v)))

# ---------- scene parameters ----------
seed = JOB['seed']; rng = random.Random(seed)
W2 = 5.0; ZS = 4.0; R = 5.0                     # half width, springline height, arch radius
base = JOB.get('rock_rgb', [112, 106, 98])
stain_rgb = (150, 92, 48)

def make_sets(joint_sets, zr):
    out = []
    for k, (ang, sp) in enumerate(joint_sets):  # trace angle in face plane (deg), spacing (m)
        a = math.radians(ang + zr.uniform(-6, 6))
        out.append({'nx': math.cos(a), 'nz': math.sin(a), 'sp': sp * zr.uniform(.85, 1.15),
                    'off': zr.uniform(0, sp), 'seed': 100 + 31 * k})
    return out

# Zones: regions of the face with their own rock-mass condition, split by wavy boundaries
# (faults, contacts, weathered bands). A job without 'zones' is a single-zone face.
zone_jobs = JOB.get('zones') or [{'code': JOB.get('rmc_code'), 'rock_rgb': base, 'joint_sets': JOB['joint_sets'],
    'block_relief_m': JOB.get('block_relief_m', .16), 'facet_scale': JOB.get('facet_scale', 1.), 'crush': JOB.get('crush', 0.),
    'weathering': JOB.get('weathering', 0.), 'stained_joint_fraction': JOB.get('stained_joint_fraction', .3),
    'joint_dark': JOB.get('joint_dark', 1.), 'facet_dark': JOB.get('facet_dark', .45), 'wet_height_m': JOB.get('wet_height_m', 1.6),
    'grain': JOB.get('grain', .10), 'weathered_rgb': JOB.get('weathered_rgb', [150, 108, 70])}]
ZONES = []
for zi, zj in enumerate(zone_jobs):
    ZONES.append({'sets': make_sets(zj['joint_sets'], random.Random(seed * 13 + zi)), 'rgb': zj['rock_rgb'],
                  'relief': zj['block_relief_m'], 'facet': zj['facet_scale'], 'crush': zj['crush'], 'wz': zj['weathering'],
                  'stain': zj['stained_joint_fraction'], 'dark': zj['joint_dark'], 'fdark': zj['facet_dark'],
                  'wet': zj['wet_height_m'], 'code': zj['code'],
                  'grain': zj.get('grain', .10), 'wcol': zj.get('weathered_rgb', [150, 108, 70])})
BOUNDS = JOB.get('boundaries', [])

def zone_of(x, z):
    """Zone index and distance (m) to the nearest boundary trace."""
    zi = 0; near = 99.
    for bi, b in enumerate(BOUNDS):
        a = math.radians(b['angle_deg']); nx_, nz_ = math.cos(a), math.sin(a)
        bs = seed + 600 + 37 * bi                       # each boundary wiggles independently
        wp = b.get('warp_m', 0.)                        # domain warp bends the trace over metres
        wx = x + wp * fbm(x * .3, z * .3, bs + 1); wz = z + wp * fbm(x * .3, z * .3, bs + 2)
        along = -nz_ * wx + nx_ * (wz - 4.3)
        wav = b.get('wav_m', .3)
        sd = nx_ * wx + nz_ * (wz - 4.3) - b['offset_m'] + wav * fbm(along * .22, 3, bs) + .4 * wav * fbm(along * .8, 5, bs + 3)
        rough = b.get('rough_m', 0.)
        if rough: sd += rough * fbm(along * 2.5, 7, bs + 4, 3)   # ragged edge at ~0.3 m
        jag = b.get('jag_m', 0.)
        if jag:  # en-echelon steps: short segments offset sideways, as when the fault follows joints
            u = along / b.get('jag_len_m', .7); k = int(math.floor(u)); f = u - k
            sd += jag * ((f - .5) * b.get('jag_ramp', .6) + (h2(k, bi, bs) - .5))
        if sd > 0: zi += 1
        near = min(near, abs(sd))
    return min(zi, len(ZONES) - 1), near

def inside_profile(x, z, grow=0.):
    if z < -grow: return False
    if z <= ZS: return abs(x) <= W2 + grow
    return x * x + (z - ZS) ** 2 <= (R + grow) ** 2

def voronoi(x, z, cell, s):
    """Nearest feature point of a jittered grid: id, offset to its centre, distance gap to second nearest."""
    gx, gz = x / cell, z / cell; i0 = int(math.floor(gx)); j0 = int(math.floor(gz))
    best = (9., 0, 0, 0., 0.); second = 9.
    for i in range(i0 - 1, i0 + 2):
        for j in range(j0 - 1, j0 + 2):
            fx = i + h2(i, j, s); fz = j + h2(i, j, s + 1)
            d2 = (gx - fx) ** 2 + (gz - fz) ** 2
            if d2 < best[0]: second = best[0]; best = (d2, i, j, (gx - fx) * cell, (gz - fz) * cell)
            elif d2 < second: second = d2
    return best[1], best[2], best[3], best[4], (math.sqrt(second) - math.sqrt(best[0])) * cell

def joint_field(x, z, P):
    """Relief toward the camera, joint darkening, iron staining and block tone at face point (x, z)."""
    key = []; d = 0.; dark = 0.; stain = 0.
    for k, js in enumerate(P['sets']):
        along = -js['nz'] * x + js['nx'] * z
        t = (js['nx'] * x + js['nz'] * z) + .12 * fbm(along * .35, k, js['seed']) + js['off']
        q = t / js['sp']; idx = int(math.floor(q))
        # irregular spacing: shift each joint by up to 35 % of the spacing
        lo = idx + .35 * (h2(idx, k, seed + 5) - .5); hi = idx + 1 + .35 * (h2(idx + 1, k, seed + 5) - .5)
        if q < lo: idx -= 1; hi = lo; lo = idx + .35 * (h2(idx, k, seed + 5) - .5)
        elif q >= hi: idx += 1; lo = hi; hi = idx + 1 + .35 * (h2(idx + 1, k, seed + 5) - .5)
        u = (q - lo) / (hi - lo); dist = js['sp'] * min(q - lo, hi - q)
        near = lo if q - lo < hi - q else hi; jn = idx if near == lo else idx + 1
        persist = max(0., min(1., .5 + 2.2 * fbm(along * .45, jn * 1.7, js['seed'] + 9)))  # non-persistent traces
        key.append(idx)
        dark = max(dark, persist * math.exp(-(dist / (.012 + .02 * h2(jn, k, seed + 6))) ** 2))
        if h2(jn, k, seed + 7) < P['stain']:
            stain = max(stain, persist * math.exp(-(dist / .16) ** 2) * (.5 + .6 * fbm(x * 2, z * 2, 55)))
        d += (h2(idx, k, seed) - .5) * .05 * persist + (h2(idx, k, seed + 3) - .5) * .5 * (u - .5) * js['sp'] * .4
    bk = h2(key[0] * 7 + key[1], key[2] if len(key) > 2 else 0, seed + 11)
    d += (bk - .5) * P['relief']
    # blast-fractured facets: each Voronoi cell is a tilted plane, two scales
    fs = P['facet']; cr = P['crush']
    scales = [(.55, .07 * fs, .55 * fs, 300, P['fdark']), (.22, .025 * fs, .45 * fs, 400, 0.)]
    if cr > 0: scales.append((.09, .02 * cr, .5 * cr, 500, .5 * cr))  # crushed / closely fractured rock
    for cell, amp, tilt, s2, gd in scales:
        ci, cj, ox, oz, gap = voronoi(x, z, cell, seed + s2)
        d += (h2(ci, cj, s2) - .5) * amp + ((h2(ci, cj, s2 + 2) - .5) * ox + (h2(ci, cj, s2 + 3) - .5) * oz) * tilt
        if gd > 0: dark = max(dark, gd * math.exp(-(gap / .012) ** 2))
    return d, dark * P['dark'], stain, bk

# ---------- object bookkeeping ----------
for obj in list(doc.Objects.GetObjectList(Rhino.DocObjects.ObjectEnumeratorSettings())):
    if obj.Attributes.GetUserString('generator') == TAG: doc.Objects.Delete(obj.Id, True)
for i in reversed(range(doc.Lights.Count)):
    if doc.Lights[i].Name and doc.Lights[i].Name.startswith(TAG): doc.Lights.Delete(i, True)
entries = []

def add(mesh, cls, name):
    mesh.Normals.ComputeNormals(); mesh.Compact()
    attr = Rhino.DocObjects.ObjectAttributes(); attr.Name = name
    attr.SetUserString('generator', TAG); attr.SetUserString('label_class', cls)
    oid = doc.Objects.AddMesh(mesh, attr)
    if oid == System.Guid.Empty: raise RuntimeError('mesh insertion failed: ' + name)
    entries.append({'id': oid, 'cls': cls, 'mesh': mesh.DuplicateMesh(), 'name': name})

def grid_zoned(nu, nv, fn):
    """Like grid(), but fn also returns a zone; quads go to one mesh per zone (by their first corner)."""
    m = Rhino.Geometry.Mesh(); zv = []
    for j in range(nv):
        for i in range(nu):
            p, c, zi = fn(i, j); zv.append(zi)
            m.Vertices.Add(p[0], p[1], p[2]); m.VertexColors.Add(Color.FromArgb(clamp(c[0]), clamp(c[1]), clamp(c[2])))
    meshes = {}
    for j in range(nv - 1):
        for i in range(nu - 1):
            a = j * nu + i
            if zv[a] not in meshes: meshes[zv[a]] = m.DuplicateMesh(); meshes[zv[a]].Faces.Clear()
            meshes[zv[a]].Faces.AddFace(a, a + 1, a + nu + 1, a + nu)
    return meshes

def grid(nu, nv, fn):
    m = Rhino.Geometry.Mesh()
    for j in range(nv):
        for i in range(nu):
            p, c = fn(i, j)
            m.Vertices.Add(p[0], p[1], p[2]); m.VertexColors.Add(Color.FromArgb(clamp(c[0]), clamp(c[1]), clamp(c[2])))
    for j in range(nv - 1):
        for i in range(nu - 1):
            a = j * nu + i; m.Faces.AddFace(a, a + 1, a + nu + 1, a + nu)
    return m

def rock_colour(x, z, dark, stain, bk, wet, P):
    # P['grain']: crystal / grain speckle contrast, the main control on fine image texture
    tone = 1 + (bk - .5) * .22 + .18 * fbm(x * 1.3, z * 1.3, seed + 40) + P['grain'] * (h2(int(x * 60), int(z * 60), seed) - .5)
    c = [v * tone for v in P['rgb']]
    wz = P['wz']
    if wz > 0:  # blend toward decomposed, iron-stained soil-like rock
        w = max(0., min(1., wz * (.75 + .6 * fbm(x * .5, z * .5, seed + 45))))
        wc = [v * (1 + .15 * fbm(x * 2.5, z * 2.5, seed + 46)) for v in P['wcol']]
        c = [c[i] * (1 - w) + wc[i] * w for i in range(3)]
    c = [c[i] * (1 - stain * .55) + stain_rgb[i] * stain * .55 for i in range(3)]
    c = [v * (1 - .48 * dark) for v in c]
    if wet > 0: c = [c[0] * (1 - .28 * wet), c[1] * (1 - .26 * wet), c[2] * (1 - .2 * wet)]
    return c

# ---------- profile helpers ----------
LEN = 2 * ZS + math.pi * R
def profile(s):
    """Point and outward normal on the excavation line, s from right foot to left foot."""
    if s <= ZS: return (W2, s), (1., 0.)
    if s >= ZS + math.pi * R: return (-W2, LEN - s), (-1., 0.)
    a = (s - ZS) / R
    return (R * math.cos(a), ZS + R * math.sin(a)), (math.cos(a), math.sin(a))

# ---------- 1. rock face ----------
step = JOB.get('face_grid_m', .035)
nx = int(2 * (W2 + .45) / step) + 1; nz = int((ZS + R + .45) / step) + 1
wet_seed = seed + 90
def face_pt(i, j):
    x = -W2 - .45 + i * step; z = j * step
    zi, near = zone_of(x, z); P = ZONES[zi]
    d, dark, stain, bk = joint_field(x, z, P)
    gw = max([b.get('gouge_m', 0.) for b in BOUNDS] + [0.])
    gouge = math.exp(-(near / gw) ** 2) if gw > 0 else 0.
    d = d * (1 - gouge) - .12 * gouge  # recessed clay gouge seam along the boundary
    r2 = (x / W2) ** 2 * .55 + ((z - 4.3) / 4.6) ** 2 * .45
    d += .32 * (1 - r2) + .05 * fbm(x * 1.6, z * 1.6, seed + 20) + .012 * fbm(x * 9, z * 9, seed + 21, 2)
    we = P['wet']; wet = max(0., min(1., (we - z) / we + .9 * fbm(x * .45, z * .45, wet_seed) + .8 * gouge))
    c = rock_colour(x, z, dark, stain, bk, wet, P)
    if gouge > 0: c = [c[k] * (1 - .55 * gouge) + [70, 60, 50][k] * .55 * gouge for k in range(3)]
    return (x, -d, z), c, zi
for zi, m in sorted(grid_zoned(nx, nz, face_pt).items()): add(m, 'rock%d' % zi, 'rock_face_zone%d' % zi)

# ---------- 2. unsupported rock strip between face and shotcrete ----------
ns = int(LEN / .04) + 1; ny = int(2.2 / .04) + 1
def strip_pt(i, j):
    s = i * LEN / (ns - 1); y = -1.9 + j * .04
    (px, pz), (qx, qz) = profile(s)
    r = .14 + .07 * fbm(s * 1.2, y * 1.2, seed + 60) + .02 * fbm(s * 7, y * 7, seed + 61, 2)
    wet = max(0., (1.2 - pz) / 1.2)
    zi, _ = zone_of(px, pz); P = ZONES[zi]
    _, dark, stain, bk = joint_field(px + .3 * y, pz, P)
    return (px + qx * r, y, pz + qz * r), rock_colour(px, pz + y, dark * .25, stain * .3, bk, wet, P), zi
for zi, m in sorted(grid_zoned(ns, ny, strip_pt).items()): add(m, 'rock%d' % zi, 'rock_perimeter_zone%d' % zi)

# ---------- 3. shotcrete lining with ragged front edge ----------
sc = JOB.get('shotcrete_rgb', [176, 170, 158])
def lining_front(s): return -1.45 - .35 * (fbm(s * .9, 0, seed + 70) + .5) - .12 * fbm(s * 5, 1, seed + 71, 2)
nsl = int(LEN / .05) + 1; nyl = 120
def lining_pt(i, j):
    s = i * LEN / (nsl - 1); f = lining_front(s); y = -16 + (f + 16) * j / (nyl - 1.)
    (px, pz), (qx, qz) = profile(s)
    r = .015 * fbm(s * 4, y * 4, seed + 80) + .006 * fbm(s * 15, y * 15, seed + 81, 2)
    tone = 1 + .12 * fbm(s * .8, y * .8, seed + 82) + .05 * (h2(int(s * 50), int(y * 50), seed + 83) - .5)
    wet = max(0., (1. - pz) / 1. + .6 * fbm(s * .5, y * .3, seed + 84))
    c = [v * tone * (1 - .35 * min(1, wet)) for v in sc]
    return (px - qx * (.0 - r), y, pz - qz * (.0 - r)), c
add(grid(nsl, nyl, lining_pt), 'shotcrete', 'shotcrete_lining')
def lip_pt(i, j):  # front end of the shotcrete layer, 0 -> 0.22 m thick
    s = i * LEN / (nsl - 1); f = lining_front(s)
    (px, pz), (qx, qz) = profile(s); r = .24 * j / 3.
    tone = .82 + .1 * fbm(s * 3, j, seed + 85)
    return (px + qx * r, f + .02 * j, pz + qz * r), [v * tone for v in sc]
add(grid(nsl, 4, lip_pt), 'shotcrete', 'shotcrete_edge')

# ---------- 4. floor and muck pile ----------
mud = JOB.get('floor_rgb', [92, 80, 66])
def floor_pt(i, j):
    x = -W2 + i * .1; y = -16 + j * .1
    t = 1 + .25 * fbm(x * .6, y * .6, seed + 95) + .08 * fbm(x * 6, y * 6, seed + 96, 2)
    return (x, y, .01 * fbm(x * 3, y * 3, seed + 97)), [v * t for v in mud]
add(grid(101, 165, floor_pt), 'other', 'floor')

def muck_h(x, y):
    b = 1 - (x / 5.8) ** 2 - ((y - .2) / 4.0) ** 2
    return max(0., b) ** .8 * JOB.get('muck_height_m', 1.5) + .12 * fbm(x * 1.4, y * 1.4, seed + 100) * max(0., b)
def muck_pt(i, j):
    x = -W2 + i * .05; y = -4.4 + j * .05
    t = .75 + .3 * fbm(x * 3, y * 3, seed + 101) + .15 * (h2(i, j, seed + 102) - .5)
    return (x, y, muck_h(x, y)), [base[0] * t * .9, base[1] * t * .88, base[2] * t * .86]
add(grid(201, 93, muck_pt), 'other', 'muck_pile')

frag = Rhino.Geometry.Mesh()
for k in range(JOB.get('fragments', 700)):
    x = rng.uniform(-4.6, 4.6); y = rng.uniform(-3.8, 0.)
    if muck_h(x, y) < .05: continue
    a = rng.uniform(.05, .2) * (2.2 if rng.random() < .1 else 1)
    box = Rhino.Geometry.Mesh.CreateIcoSphere(Rhino.Geometry.Sphere(Rhino.Geometry.Point3d.Origin, a), 1)
    sx, sy, sz = rng.uniform(.8, 1.3), rng.uniform(.6, 1.1), rng.uniform(.35, .7)
    for vi in range(box.Vertices.Count):
        v = box.Vertices[vi]; f = rng.uniform(.7, 1.15); box.Vertices.SetVertex(vi, v.X * sx * f, v.Y * sy * f, v.Z * sz * f)
    box.Unweld(0, True)
    box.Transform(Rhino.Geometry.Transform.Rotation(rng.uniform(0, 6.3), Rhino.Geometry.Vector3d.ZAxis, Rhino.Geometry.Point3d.Origin))
    box.Transform(Rhino.Geometry.Transform.Rotation(rng.uniform(-.5, .5), Rhino.Geometry.Vector3d.XAxis, Rhino.Geometry.Point3d.Origin))
    box.Translate(x, y, muck_h(x, y) + a * .2)
    t = rng.uniform(.42, .85)
    box.VertexColors.CreateMonotoneMesh(Color.FromArgb(clamp(base[0] * t), clamp(base[1] * t), clamp(base[2] * t)))
    frag.Append(box)
add(frag, 'other', 'muck_fragments')

# ---------- lights and camera ----------
def light(name, style, loc, target, intensity, angle=None):
    L = Rhino.Geometry.Light(); L.Name = TAG + '_' + name; L.IsEnabled = True; L.LightStyle = style
    L.Location = Rhino.Geometry.Point3d(*loc); L.Direction = Rhino.Geometry.Vector3d(*[target[i] - loc[i] for i in range(3)])
    L.Diffuse = Color.FromArgb(255, 246, 232); L.Intensity = intensity
    if angle: L.SpotAngleRadians = math.radians(angle); L.HotSpot = .55
    return doc.Lights.Add(L)
cam = JOB['camera']
light('key', Rhino.Geometry.LightStyle.WorldSpot, cam['key_light'], [0, 0, 4.2], JOB.get('key_intensity', .7), 48)
light('fill', Rhino.Geometry.LightStyle.WorldPoint, [-3.2, -5.5, 1.6], [0, 0, 3], JOB.get('fill_intensity', .12))

def mode(name, is_id):
    m = next((d for d in Rhino.Display.DisplayModeDescription.GetDisplayModes() if d.EnglishName == name), None)
    if m is None:
        m = Rhino.Display.DisplayModeDescription.GetDisplayMode(Rhino.Display.DisplayModeDescription.CopyDisplayMode(Rhino.Display.DisplayModeDescription.ShadedId, name))
    a = m.DisplayAttributes
    for p in ['ShowSurfaceEdges', 'ShowMeshEdges', 'ShowMeshNakedEdges', 'ShowIsoCurves', 'ShowCurves', 'ShowLights', 'ShowPoints',
              'ShowText', 'ShowAnnotations', 'CustomGroundPlaneOn', 'HighlightSurfaces', 'PreProcessColors', 'PreProcessTextures']:
        setattr(a, p, False)
    a.ShadowsOn = not is_id; a.CastShadows = not is_id
    a.IgnoreHighlights = True; a.CullBackfaces = False; a.UseBackfaceMaterial = False
    a.UseAssignedObjectMaterial = False; a.UseCustomObjectMaterial = False; a.UseCustomObjectColor = False
    a.ShadingEnabled = True; a.ShadeVertexColors = True
    a.MeshSpecificAttributes.ShowMeshWires = False; a.MeshSpecificAttributes.ShowMeshVertices = False
    a.LightingScheme = getattr(Rhino.Display.DisplayPipelineAttributes.LightingSchema, 'None') if is_id else Rhino.Display.DisplayPipelineAttributes.LightingSchema.SceneLighting
    a.AmbientLightingColor = Color.FromArgb(*JOB.get('ambient_rgb', [16, 15, 14])); a.FrontMaterialShine = 0
    a.SetFill(Color.Black if is_id else Color.FromArgb(12, 11, 10))
    va = a.ViewSpecificAttributes
    va.UseDocumentGrid = False; va.DrawGrid = False; va.DrawGridAxes = False; va.DrawWorldAxes = False; va.DrawZAxis = False
    Rhino.Display.DisplayModeDescription.UpdateDisplayMode(m)
    return Rhino.Display.DisplayModeDescription.GetDisplayMode(m.Id)
rgb_mode = mode('TunnelFace_RGB', False); id_mode = mode('TunnelFace_ID', True)
vp.ConstructionGridVisible = False; vp.ConstructionAxesVisible = False; vp.WorldAxesVisible = False
vp.ChangeToPerspectiveProjection(True, cam['lens_mm'])
vp.SetCameraLocations(Rhino.Geometry.Point3d(*cam['target']), Rhino.Geometry.Point3d(*cam['location']))
vp.CameraUp = Rhino.Geometry.Vector3d.ZAxis; vp.Camera35mmLensLength = cam['lens_mm']
view.Redraw()

out = os.path.join(ROOT, 'render'); W, H = JOB['capture_size']
if not os.path.isdir(out): os.makedirs(out)
def capture(fname, display):
    vp.DisplayMode = display; view.Redraw(); Rhino.RhinoApp.Wait()
    bmp = view.CaptureToBitmap(System.Drawing.Size(W, H), display)
    if bmp is None: raise RuntimeError('view capture failed')
    bmp.Save(os.path.join(out, fname), ImageFormat.Png); bmp.Dispose()

name = 'tunnel_face_%03d' % seed
capture(name + '_rgb.png', rgb_mode)
id_rgb = {'rock0': (0, 255, 0), 'rock1': (255, 0, 0), 'rock2': (255, 0, 255), 'shotcrete': (0, 0, 255), 'other': (0, 0, 0)}
for e in entries:  # same geometry and camera, only vertex colours change
    m = e['mesh'].DuplicateMesh(); m.VertexColors.CreateMonotoneMesh(Color.FromArgb(*id_rgb[e['cls']]))
    doc.Objects.Replace(e['id'], m)
capture(name + '_id.png', id_mode)
for e in entries: doc.Objects.Replace(e['id'], e['mesh'])
vp.DisplayMode = rgb_mode; view.Redraw()
f = open(os.path.join(out, name + '_scene.json'), 'w')
json.dump({'seed': seed, 'name': name, 'job': JOB, 'zones': [z['code'] for z in ZONES], 'objects': [{'name': e['name'], 'class': e['cls']} for e in entries],
           'rhino_version': str(Rhino.RhinoApp.Version)}, f, indent=2); f.close()
print(json.dumps({'status': 'ok', 'name': name, 'objects': len(entries)}))
