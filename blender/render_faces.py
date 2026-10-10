"""Render tunnel faces in Blender (Cycles), one or many jobs per Blender start.

Run headless:
  blender -b --factory-startup -P blender/render_faces.py -- --jobs jobs.json --out OUT [--device GPU] [--samples 32]

jobs.json is a list of job dicts (as written by scripts/batch_tunnel_faces.py).
For each job the scene geometry comes from tunnelgen.geometry.build_scene, so it
matches the Rhino generator. Writes OUT/render/<name>_rgb.png (the shaded view),
OUT/render/<name>_index.npz (exact per-pixel object class index, no anti-aliasing, top row first)
and OUT/render/<name>_scene.json (index -> class map, zones, timing).
"""
import argparse, json, math, os, sys, time
import numpy as np
import bpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tunnelgen.geometry import build_scene  # noqa: E402

CLASS_INDEX = {'rock0': 1, 'rock1': 2, 'rock2': 3, 'shotcrete': 4, 'other': 5}   # 0 = background


def args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='GPU', choices=['GPU', 'CPU'])
    ap.add_argument('--samples', type=int, default=32)
    ap.add_argument('--scale', type=float, default=1.0, help='resolution scale for quick tests')
    ap.add_argument('--key-watts', type=float, default=6000., help='key spot power at key_intensity 0.7')
    ap.add_argument('--textures', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'textures'),
                    help='bank of real-photo detail tiles (make_texture_bank.py); missing folder = procedural grain only')
    ap.add_argument('--denoiser', default='OPTIX', choices=['OPTIX', 'OPENIMAGEDENOISE', 'NONE'],
                    help='OPTIX on NVIDIA; OpenImageDenoise was ~20 s/frame on an RTX 5090 (Blender 4.5)')
    return ap.parse_args(argv)


def setup_render(a):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    cy = sc.cycles
    cy.samples = a.samples; cy.use_adaptive_sampling = True; cy.use_denoising = a.denoiser != 'NONE'
    cy.max_bounces = 4; cy.diffuse_bounces = 2; cy.glossy_bounces = 1; cy.transmission_bounces = 0
    cy.device = 'CPU'
    if a.device == 'GPU':
        prefs = bpy.context.preferences.addons['cycles'].preferences
        for backend in ('OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI'):
            try:
                prefs.compute_device_type = backend
            except TypeError:
                continue
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == backend]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type == backend
                cy.device = 'GPU'
                print('Cycles device:', backend, [d.name for d in gpus], flush=True)
                break
        if cy.device != 'GPU':
            print('No GPU backend found; rendering on CPU', flush=True)
    if a.denoiser != 'NONE':
        if a.denoiser == 'OPTIX' and cy.device != 'GPU':
            a.denoiser = 'OPENIMAGEDENOISE'
        cy.denoiser = a.denoiser
    sc.view_settings.view_transform = 'Standard'; sc.view_settings.look = 'None'
    sc.render.image_settings.file_format = 'PNG'; sc.render.image_settings.color_mode = 'RGB'
    sc.render.film_transparent = False
    vl = sc.view_layers[0]; vl.use_pass_object_index = True
    # compositor: index pass -> float EXR, written next to the RGB image
    sc.use_nodes = True
    tree = sc.node_tree; tree.nodes.clear()
    rl = tree.nodes.new('CompositorNodeRLayers')
    comp = tree.nodes.new('CompositorNodeComposite'); tree.links.new(rl.outputs['Image'], comp.inputs['Image'])
    fo = tree.nodes.new('CompositorNodeOutputFile')
    fo.format.file_format = 'OPEN_EXR'; fo.format.color_depth = '32'; fo.format.color_mode = 'BW'
    tree.links.new(rl.outputs['IndexOB'], fo.inputs[0])
    return fo


def clear_scene():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
        for b in list(coll):
            coll.remove(b)


def vertex_colour_material():
    """Vertex colour x procedural grain (per-object 'grain', 'grain_scale'), micro-bump ('bump') and wet gloss ('Wet')."""
    mat = bpy.data.materials.new('rock_vertex_colour'); mat.use_nodes = True
    nt = mat.node_tree; N, L = nt.nodes, nt.links; bsdf = N['Principled BSDF']

    def attr(name, kind):
        a = N.new('ShaderNodeAttribute'); a.attribute_name = name; a.attribute_type = kind; return a

    def math_node(op, a=None, b=None):
        m = N.new('ShaderNodeMath'); m.operation = op
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                m.inputs[i].default_value = v
            else:
                L.new(v, m.inputs[i])
        return m.outputs[0]

    col = N.new('ShaderNodeVertexColor'); col.layer_name = 'Col'
    wet = attr('Wet', 'GEOMETRY').outputs['Fac']
    grain, gscale, bump = (attr(n, 'OBJECT').outputs['Fac'] for n in ('grain', 'grain_scale', 'bump'))
    tc = N.new('ShaderNodeTexCoord')
    noise = N.new('ShaderNodeTexNoise'); noise.inputs['Detail'].default_value = 3.; noise.inputs['Roughness'].default_value = .65
    L.new(tc.outputs['Object'], noise.inputs['Vector']); L.new(gscale, noise.inputs['Scale'])
    # tone = 1 + 2 * grain * (noise - 0.5) - 0.35 * wet
    tone = math_node('ADD', math_node('MULTIPLY', math_node('MULTIPLY', math_node('SUBTRACT', noise.outputs['Fac'], .5), grain), 2.), 1.)
    tone = math_node('SUBTRACT', tone, math_node('MULTIPLY', wet, .35))
    mix = N.new('ShaderNodeVectorMath'); mix.operation = 'SCALE'
    L.new(col.outputs['Color'], mix.inputs[0]); L.new(tone, mix.inputs['Scale'])
    L.new(mix.outputs['Vector'], bsdf.inputs['Base Color'])
    bn = N.new('ShaderNodeBump'); bn.inputs['Distance'].default_value = .004
    L.new(noise.outputs['Fac'], bn.inputs['Height']); L.new(math_node('MULTIPLY', bump, .6), bn.inputs['Strength'])
    L.new(bn.outputs['Normal'], bsdf.inputs['Normal'])
    L.new(math_node('SUBTRACT', .88, math_node('MULTIPLY', wet, .62)), bsdf.inputs['Roughness'])
    if 'Specular IOR Level' in bsdf.inputs:
        L.new(math_node('ADD', .25, math_node('MULTIPLY', wet, .4)), bsdf.inputs['Specular IOR Level'])
    return mat


def rock_material(name, tiles, tile_m, rng, crack_m, crack_strength, detail_strength):
    """Vertex colour x two real-photo detail tiles (box-projected in object space at output-pixel scale)
    x class-independent Voronoi blast cracks, with bump from the tiles and wet gloss from the 'Wet' attribute."""
    mat = bpy.data.materials.new(name); mat.use_nodes = True
    nt = mat.node_tree; N, L = nt.nodes, nt.links; bsdf = N['Principled BSDF']

    def attr(n, kind):
        a = N.new('ShaderNodeAttribute'); a.attribute_name = n; a.attribute_type = kind; return a

    def M(op, a=None, b=None):
        m = N.new('ShaderNodeMath'); m.operation = op
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                m.inputs[i].default_value = v
            else:
                L.new(v, m.inputs[i])
        return m.outputs[0]

    col = N.new('ShaderNodeVertexColor'); col.layer_name = 'Col'
    wet = attr('Wet', 'GEOMETRY').outputs['Fac']
    tc = N.new('ShaderNodeTexCoord')

    def tile_node(path, scale_m):
        mp = N.new('ShaderNodeMapping'); mp.vector_type = 'POINT'
        mp.inputs['Rotation'].default_value = (rng.uniform(0, 6.28), rng.uniform(0, 6.28), rng.uniform(0, 6.28))
        mp.inputs['Scale'].default_value = (1. / scale_m,) * 3
        L.new(tc.outputs['Object'], mp.inputs['Vector'])
        im = N.new('ShaderNodeTexImage'); im.image = bpy.data.images.load(path, check_existing=True)
        im.image.colorspace_settings.name = 'Non-Color'; im.projection = 'BOX'; im.projection_blend = .3; im.extension = 'REPEAT'
        L.new(mp.outputs['Vector'], im.inputs['Vector'])
        return M('MULTIPLY', im.outputs['Color'], 255. / 128.)          # stored value = 128 * detail  ->  detail (mean 1)

    if tiles:
        t1 = tile_node(tiles[0], tile_m); t2 = tile_node(tiles[1], tile_m * rng.uniform(.7, 1.4))
        mask = N.new('ShaderNodeTexNoise'); mask.inputs['Scale'].default_value = 1.2 / max(tile_m, .3); mask.inputs['Detail'].default_value = 2.
        L.new(tc.outputs['Object'], mask.inputs['Vector'])
        mixn = N.new('ShaderNodeMix'); mixn.data_type = 'FLOAT'
        L.new(mask.outputs['Fac'], mixn.inputs['Factor']); L.new(t1, mixn.inputs[2]); L.new(t2, mixn.inputs[3])
        detail = mixn.outputs[0]
    else:
        noise = N.new('ShaderNodeTexNoise'); noise.inputs['Scale'].default_value = 150.; noise.inputs['Detail'].default_value = 3.
        L.new(tc.outputs['Object'], noise.inputs['Vector'])
        detail = M('ADD', M('MULTIPLY', M('SUBTRACT', noise.outputs['Fac'], .5), .6), 1.)
    # tone = 1 + k (detail - 1): k sets the fine-texture contrast
    tone = M('ADD', M('MULTIPLY', M('SUBTRACT', detail, 1.), detail_strength), 1.)
    # blast cracks: dark Voronoi cell edges at a physical scale, the same for every class
    vor = N.new('ShaderNodeTexVoronoi'); vor.feature = 'DISTANCE_TO_EDGE'; vor.inputs['Scale'].default_value = 1. / crack_m
    vor.inputs['Randomness'].default_value = .9
    L.new(tc.outputs['Object'], vor.inputs['Vector'])
    crack = M('SMOOTH_MIN', M('DIVIDE', vor.outputs['Distance'], crack_m * .15), 1.)   # 0 on the edge -> 1 inside
    tone = M('MULTIPLY', tone, M('ADD', M('MULTIPLY', crack, crack_strength), 1. - crack_strength))
    tone = M('SUBTRACT', tone, M('MULTIPLY', wet, .35))
    mix = N.new('ShaderNodeVectorMath'); mix.operation = 'SCALE'
    L.new(col.outputs['Color'], mix.inputs[0]); L.new(tone, mix.inputs['Scale'])
    L.new(mix.outputs['Vector'], bsdf.inputs['Base Color'])
    bn = N.new('ShaderNodeBump'); bn.inputs['Distance'].default_value = .02; bn.inputs['Strength'].default_value = .8
    L.new(M('MULTIPLY', detail, crack), bn.inputs['Height']); L.new(bn.outputs['Normal'], bsdf.inputs['Normal'])
    L.new(M('SUBTRACT', .9, M('MULTIPLY', wet, .65)), bsdf.inputs['Roughness'])
    if 'Specular IOR Level' in bsdf.inputs:
        L.new(M('ADD', .2, M('MULTIPLY', wet, .5)), bsdf.inputs['Specular IOR Level'])
    return mat


def add_mesh(m, mat, zone_params=None):
    v, f = m['verts'], m['faces']
    me = bpy.data.meshes.new(m['name'])
    me.vertices.add(len(v)); me.vertices.foreach_set('co', v.ravel())
    k = f.shape[1]
    me.loops.add(f.size); me.loops.foreach_set('vertex_index', f.ravel())
    me.polygons.add(len(f))
    me.polygons.foreach_set('loop_start', np.arange(0, f.size, k, dtype=np.int32))
    me.polygons.foreach_set('loop_total', np.full(len(f), k, np.int32))
    me.update(calc_edges=True); me.validate(clean_customdata=False)
    col = me.color_attributes.new('Col', 'BYTE_COLOR', 'POINT')
    rgba = np.ones((len(v), 4), np.float32); rgba[:, :3] = m['colours'] / 255.
    col.data.foreach_set('color_srgb', rgba.ravel())
    me.polygons.foreach_set('use_smooth', np.full(len(f), not m['flat'], bool))
    wet = me.attributes.new('Wet', 'FLOAT', 'POINT'); wet.data.foreach_set('value', m.get('wet', np.zeros(len(v), np.float32)))
    me.materials.append(mat)
    ob = bpy.data.objects.new(m['name'], me); ob.pass_index = CLASS_INDEX[m['cls']]
    zp = zone_params[m['zone']] if zone_params and m.get('zone') is not None else {}
    ob['grain'] = float(zp.get('grain', 0.) if zp.get('grain_scale') else 0.)   # old jobs: grain is in the vertex colours
    ob['grain_scale'] = float(zp.get('grain_scale', 0.) or 150.); ob['bump'] = float(zp.get('bump', 0.))
    bpy.context.scene.collection.objects.link(ob)
    return ob


def look_at(ob, loc, target):
    ob.location = loc
    d = np.array(target, float) - np.array(loc, float)
    ob.rotation_euler = (math.atan2(math.hypot(d[0], d[1]), -d[2]), 0., math.atan2(d[1], d[0]) - math.pi / 2)


def kelvin_rgb(k):
    """Approximate colour of a black-body light at k kelvin (Tanner Helland fit), 0-1."""
    t = k / 100.
    r = 1. if t <= 66 else min(1., 1.292936 * (t - 60) ** -.1332047)
    g = min(1., max(0., (.39008158 * math.log(t) - .63184144) if t <= 66 else 1.12989086 * (t - 60) ** -.0755148))
    b = 1. if t >= 66 else (0. if t <= 19 else min(1., .54320679 * math.log(t - 10) - 1.19625409))
    return (r, g, b)


def add_camera_lights(S, a, W, H):
    sc = bpy.context.scene; cam = S['camera']
    cd = bpy.data.cameras.new('cam'); cd.lens = cam['lens_mm']; cd.sensor_width = 36.; cd.sensor_fit = 'HORIZONTAL'
    co = bpy.data.objects.new('cam', cd); sc.collection.objects.link(co); sc.camera = co
    look_at(co, cam['location'], cam['target'])
    sc.render.resolution_x = int(W * a.scale); sc.render.resolution_y = int(H * a.scale); sc.render.resolution_percentage = 100
    for L in S['lights']:
        ld = bpy.data.lights.new(L['name'], L['type'])
        ld.color = kelvin_rgb(S['light_kelvin']) if S.get('light_kelvin') else (1., 246 / 255., 232 / 255.)
        # inverse-square: the same exposure whether the light is 3 m or 10 m from the face
        dist = float(np.linalg.norm(np.array(L['location'], float) - np.array(cam['target'], float)))
        ld.energy = a.key_watts * L['intensity'] / .7 * (dist / 10.) ** 2 if S.get('light_kelvin') else a.key_watts * L['intensity'] / .7
        if L['type'] == 'SPOT':
            ld.spot_size = math.radians(2 * L['spot_deg']); ld.spot_blend = S.get('spot_blend', .45)
        ld.shadow_soft_size = S.get('shadow_soft', .15)
        lo = bpy.data.objects.new(L['name'], ld); sc.collection.objects.link(lo)
        look_at(lo, L['location'], L['target'])
    world = bpy.data.worlds.get('World') or bpy.data.worlds.new('World'); sc.world = world
    world.use_nodes = True
    amb = [c / 255. for c in S['ambient_rgb']]
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (*amb, 1.)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = 1.
    sc.view_settings.exposure = float(S.get('exposure', 0.))


def main():
    a = args()
    jobs = json.load(open(a.jobs))
    out = os.path.join(a.out, 'render'); os.makedirs(out, exist_ok=True)
    fo = setup_render(a)
    for job in jobs:
        name = 'tunnel_face_%03d' % job['seed']
        t0 = time.time()
        clear_scene()
        S = build_scene(job); t1 = time.time()
        mat = vertex_colour_material()
        rng = np.random.default_rng(job['seed'] + 5)
        bank = sorted(f for f in os.listdir(a.textures) if f.endswith('.png')) if os.path.isdir(a.textures) else []
        cam = job['camera']; dist = float(np.linalg.norm(np.array(cam['location']) - np.array(cam['target'])))
        gsd = 36. / cam['lens_mm'] * dist / job['capture_size'][0]            # metres per render pixel at the target
        tile_m = 256 * gsd                                                   # one tile pixel = one render pixel
        crack_m = max(.05, 6 * gsd); ds = job.get('detail_strength', 1.2); cs = job.get('crack_strength', .45)
        mats = {}
        pick = lambda: [os.path.join(a.textures, bank[int(rng.integers(len(bank)))]) for _ in range(2)]
        for m in S['meshes']:
            if not bank:
                add_mesh(m, mat, S.get('zone_params')); continue
            # real-photo detail on every surface, so texture alone does not separate rock from lining or muck
            key = m['zone'] if m['cls'].startswith('rock') else m['cls']
            if key not in mats:
                if m['cls'].startswith('rock'):
                    mats[key] = rock_material('rock_zone%s' % key, pick(), tile_m * job.get('tile_scale', 1.), rng, crack_m, cs, ds)
                elif m['cls'] == 'shotcrete':
                    mats[key] = rock_material('shotcrete', pick(), tile_m * 1.3, rng, crack_m * 2, .1, ds * .55)
                else:
                    mats[key] = rock_material('other', pick(), tile_m * .8, rng, crack_m, .3, ds * .9)
            add_mesh(m, mats[key], S.get('zone_params'))
        W, H = job['capture_size']
        add_camera_lights(S, a, W, H)
        sc = bpy.context.scene
        sc.render.filepath = os.path.join(out, name + '_rgb.png')
        fo.base_path = out; fo.file_slots[0].path = name + '_index_'
        sc.frame_set(1)
        bpy.ops.render.render(write_still=True)
        exr = os.path.join(out, name + '_index_0001.exr')
        im = bpy.data.images.load(exr)
        px = np.empty(im.size[0] * im.size[1] * im.channels, np.float32); im.pixels.foreach_get(px)
        idx = np.flipud(px.reshape(im.size[1], im.size[0], im.channels)[..., 0]).round().astype(np.uint8)  # top row first
        np.savez_compressed(os.path.join(out, name + '_index.npz'), index=idx)
        bpy.data.images.remove(im); os.remove(exr)
        t2 = time.time()
        json.dump({'seed': job['seed'], 'name': name, 'job': job, 'zones': S['zones'], 'zone_gsi': S.get('zone_gsi'),
                   'class_index': CLASS_INDEX, 'objects': [{'name': m['name'], 'class': m['cls']} for m in S['meshes']],
                   'renderer': 'blender %s cycles %s' % (bpy.app.version_string, sc.cycles.device),
                   'samples': a.samples, 'seconds': {'geometry': round(t1 - t0, 2), 'render': round(t2 - t1, 2)}},
                  open(os.path.join(out, name + '_scene.json'), 'w'), indent=1)
        print('RENDERED %s geometry %.1fs render %.1fs' % (name, t1 - t0, t2 - t1), flush=True)


if __name__ == '__main__':
    main()
