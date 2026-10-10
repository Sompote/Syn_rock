"""Job sampler for the foundation dataset: geology first, appearance second, labels last.

For each face it draws 1-3 zones (single rock mass, contact, fault band or dyke),
and for each zone a GSI structure class, a rock type, an ISRM weathering grade and
the parameters that follow from them; then the groundwater condition, support,
occluders, camera framing and lighting. The GSI class of each zone is the
pretraining label. An RMC code is also derived through an assumed mapping
(A best .. E worst, see rmc_code) so the same images can carry the RMC label set.
See docs/geological_patterns.md.
"""
import random

GSI = ['intact_massive', 'blocky', 'very_blocky', 'blocky_disturbed_seamy', 'disintegrated', 'laminated_sheared']

# rock types: base colour (sRGB), grain contrast and scale (1/m), typical bedding/foliation, veins, karst
ROCKS = {
    'granite':   dict(rgb=[(150, 146, 140), (128, 120, 112), (170, 160, 150)], grain=(.25, .45), gscale=(120, 220), layered=0., veins=.3, karst=0.),
    'gneiss':    dict(rgb=[(120, 118, 115), (140, 132, 122)], grain=(.2, .35), gscale=(100, 180), layered=.8, veins=.3, karst=0.),
    'schist':    dict(rgb=[(110, 112, 104), (96, 100, 96), (128, 120, 108)], grain=(.12, .25), gscale=(150, 260), layered=1., veins=.25, karst=0.),
    'sandstone': dict(rgb=[(160, 140, 112), (140, 124, 100), (170, 150, 120)], grain=(.10, .2), gscale=(200, 320), layered=.7, veins=.05, karst=0.),
    'shale':     dict(rgb=[(78, 76, 74), (92, 88, 82), (70, 72, 74)], grain=(.05, .12), gscale=(250, 400), layered=.9, veins=.1, karst=0.),
    'limestone': dict(rgb=[(168, 166, 158), (150, 150, 144), (182, 176, 164)], grain=(.06, .15), gscale=(150, 300), layered=.5, veins=.6, karst=.35),
    'basalt':    dict(rgb=[(70, 70, 72), (84, 82, 80), (62, 64, 66)], grain=(.08, .16), gscale=(250, 400), layered=0., veins=.15, karst=0.),
}
WEATHERED_RGB = [(146, 116, 88), (134, 110, 86), (154, 126, 96), (126, 102, 80)]   # desaturated: real E regions are 16 % orange, not 35 %


def u(r, lo_hi):
    return r.uniform(*lo_hi)


def structure_params(r, gsi, angles):
    """Joint sets, relief, facets and crushing for one GSI structure class."""
    a = list(angles)
    if gsi == 0:
        sets = [[a[0], u(r, (1.6, 4.))]] + [[a[k], u(r, (2.5, 6.))] for k in (1, 2)]
        p = dict(block_relief_m=u(r, (.02, .05)), facet_scale=u(r, (.4, .7)), crush=0., joint_dark=u(r, (.5, .8)), facet_dark=u(r, (.05, .15)))
    elif gsi == 1:
        sets = [[a[k], u(r, (.6, 1.5))] for k in range(3)]
        p = dict(block_relief_m=u(r, (.08, .14)), facet_scale=u(r, (.8, 1.)), crush=0., joint_dark=u(r, (.7, .95)), facet_dark=u(r, (.2, .35)))
    elif gsi == 2:
        sets = [[a[k], u(r, (.2, .6))] for k in range(4)]
        p = dict(block_relief_m=u(r, (.12, .18)), facet_scale=u(r, (1., 1.2)), crush=u(r, (0., .3)), joint_dark=u(r, (.85, 1.)), facet_dark=u(r, (.35, .5)))
    elif gsi == 3:
        sets = [[a[k], u(r, (.15, .35))] for k in range(3)]
        p = dict(block_relief_m=u(r, (.14, .2)), facet_scale=u(r, (1., 1.2)), crush=u(r, (.3, .6)), joint_dark=u(r, (.85, 1.)), facet_dark=u(r, (.4, .5)))
    elif gsi == 4:
        sets = [[a[k], u(r, (.08, .15))] for k in range(3)]
        p = dict(block_relief_m=u(r, (.15, .25)), facet_scale=u(r, (.4, .7)), crush=u(r, (.8, 1.)), joint_dark=u(r, (.4, .7)), facet_dark=u(r, (.2, .35)))
    else:
        sets = [[a[1], u(r, (.6, 1.5))], [a[2], u(r, (1., 3.))]]
        p = dict(block_relief_m=u(r, (.04, .09)), facet_scale=u(r, (.3, .6)), crush=u(r, (0., .3)), joint_dark=u(r, (.5, .8)), facet_dark=u(r, (.05, .15)))
    p['joint_sets'] = sets
    p['joint_dark'] *= .6; p['facet_dark'] *= .7     # trace darkening weakened: real edge density is class-independent
    return p


def bedding_params(r, gsi, rock, angle):
    """Bedding, lamination or foliation: always for laminated, often for disturbed, by rock type otherwise."""
    layered = ROCKS[rock]['layered']
    if gsi == 5:
        sp, fold = u(r, (.05, .12)), (u(r, (.2, 1.2)) if r.random() < .5 else 0.)
    elif gsi == 3:
        if r.random() > .8:
            return None
        sp, fold = u(r, (.1, .3)), u(r, (.4, 1.5))
    elif r.random() < layered * .6:
        sp, fold = u(r, (.15, .6)), (u(r, (.2, .8)) if r.random() < .25 else 0.)
    else:
        return None
    return dict(angle_deg=angle, spacing_m=sp, fold_amp_m=fold, fold_len_m=u(r, (3., 9.)),
                band_contrast=u(r, (.08, .25)) * (1.5 if rock in ('sandstone', 'shale', 'gneiss') else 1.),
                relief_m=u(r, (.005, .03)), dark=u(r, (.4, .9)))


def zone(r, gsi, rock, angles, weathering_grade):
    rt = ROCKS[rock]
    z = structure_params(r, gsi, angles)
    w = [0., u(r, (0., .05)), u(r, (.1, .2)), u(r, (.25, .4)), u(r, (.3, .45)), u(r, (.4, .55))][weathering_grade]
    if weathering_grade >= 4:   # soil-like: blunt facets, faint joints, lumpy relief
        z['facet_scale'] *= .5; z['joint_dark'] *= .5; z['facet_dark'] *= .5; z['crush'] = max(z['crush'], .6)
    base = list(r.choice(rt['rgb'])); f = u(r, (.85, 1.15))
    z.update(gsi=gsi, rock=rock, weathering_grade=weathering_grade,
             rock_rgb=[max(25, min(225, int(c * f) + r.randint(-4, 4))) for c in base],
             grain=u(r, rt['grain']), grain_scale=u(r, rt['gscale']), bump=u(r, (.1, .4)),
             weathering=w, weathered_rgb=list(r.choice(WEATHERED_RGB)),
             # staining on every class (real A-D regions show 2-6 % orange, E 16 %), rising with weathering
             stained_joint_fraction=min(.95, [.25, .3, .35, .45, .5, .5][weathering_grade] + u(r, (0., .15))),
             wet_height_m=u(r, (.5, 2.5)),
             bedding=bedding_params(r, gsi, rock, angles[0]),
             veins=[dict(angle_deg=u(r, (0, 180)), offset_m=u(r, (-4, 4)), width_m=u(r, (.012, .04)),
                         rgb=list(r.choice([(225, 222, 212), (205, 200, 190), (235, 232, 225)])))
                    for _ in range(r.randint(1, 6))] if r.random() < rt['veins'] else [],
             karst=[dict(x=u(r, (-4, 4)), z=u(r, (1, 8)), r=u(r, (.15, .6)), depth=u(r, (.15, .5)))
                    for _ in range(r.randint(1, 4))] if r.random() < rt['karst'] else [])
    return z


def rmc_code(z, r):
    """Assumed mapping from geology to an RMC code (A best .. E worst); the source data define no codebook."""
    letter = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 3}[z['gsi']]
    if z['weathering_grade'] >= 4:
        letter = min(4, letter + 1)
    sub = 1 + min(2, int(r.random() * 3))
    return 'RMC%s%02d' % ('ABCDE'[letter], sub)


def boundary(r, kind):
    if kind == 'contact':
        return dict(warp_m=u(r, (.8, 2.4)), wav_m=u(r, (1., 2.4)), rough_m=u(r, (.08, .2)),
                    jag_m=u(r, (.2, .5)) if r.random() < .5 else 0., jag_len_m=u(r, (.4, 1.1)), jag_ramp=u(r, (.3, 1.)), gouge_m=0.)
    if kind == 'fault':
        return dict(warp_m=u(r, (.6, 1.6)), wav_m=u(r, (.8, 1.6)), rough_m=u(r, (.1, .25)),
                    jag_m=u(r, (.2, .45)) if r.random() < .6 else 0., jag_len_m=u(r, (.4, 1.1)), jag_ramp=u(r, (.3, 1.)), gouge_m=u(r, (.05, .14)))
    return dict(warp_m=u(r, (.1, .4)), wav_m=u(r, (.1, .4)), rough_m=u(r, (.02, .06)), jag_m=0., gouge_m=0.)   # dyke: sharp, straight walls


def sample_job(seed):
    r = random.Random(seed * 2654435761 % 2 ** 32)
    angles = [u(r, (0, 180)) for _ in range(4)]
    host_rock = r.choice(list(ROCKS))
    host_gsi = r.choices(range(6), weights=[1.2, 1.4, 1.4, 1., .8, .8])[0]
    host_w = r.choices(range(6), weights=[.6, 1.2, 1., .6, .3, .2])[0]
    pattern = r.choices(['single', 'contact', 'fault_band', 'dyke'], weights=[.4, .25, .25, .1])[0]
    zones, bounds = [zone(r, host_gsi, host_rock, angles, min(5, host_w + (host_gsi == 4)))], []
    trace = u(r, (30, 150))
    if pattern == 'contact':
        other_rock = r.choice(list(ROCKS)); g = r.choices(range(6), weights=[1, 1, 1, 1, .8, .8])[0]
        zones.append(zone(r, g, other_rock, angles, r.choices(range(6), weights=[.6, 1.2, 1., .6, .3, .2])[0]))
        bounds = [dict(boundary(r, 'contact'), angle_deg=trace + 90, offset_m=u(r, (-2., 2.)))]
    elif pattern in ('fault_band', 'dyke'):
        if pattern == 'fault_band':
            band = zone(r, r.choice([3, 4, 4, 5]), host_rock, angles, min(5, host_w + r.randint(1, 3)))
            kind, width = 'fault', u(r, (.8, 3.))
        else:
            band = zone(r, r.choice([0, 1, 1, 2]), 'basalt', angles, max(0, host_w - 1))
            kind, width = 'dyke', u(r, (.5, 2.))
        side = zone(r, host_gsi if r.random() < .6 else min(5, max(0, host_gsi + r.randint(-1, 1))), host_rock, angles, host_w)
        zones += [band, side]
        o1 = u(r, (-2.5, 1.))
        bounds = [dict(boundary(r, kind), angle_deg=trace + 90 + u(r, (-8, 8)), offset_m=o1 + w) for w in (0., width)]
    for z in zones:
        z['code'] = rmc_code(z, r)

    water = r.choices(['dry', 'damp', 'wet', 'dripping', 'flowing'], weights=[.3, .3, .2, .12, .08])[0]
    closeup = r.random() < .55
    if closeup:
        dist = u(r, (3.5, 7.)); tx, tz = u(r, (-3., 3.)), u(r, (2., 6.5)); lens = u(r, (24., 45.))
    else:
        dist = u(r, (8.5, 11.)); tx, tz = u(r, (-.4, .4)), u(r, (4.1, 4.7)); lens = u(r, (18., 26.))
    cx = tx + u(r, (-1.2, 1.2)); cz = u(r, (1.8, 3.8))
    flash = r.random() < .5
    key = [cx + u(r, (-.3, .3)), -dist + .3, cz + u(r, (-.2, .3))] if flash else [cx + u(r, (-2.5, 2.5)), -dist + u(r, (1., 3.)), u(r, (3., 5.5))]
    return {
        'seed': seed, 'generator': 'tunnelgen.sampler', 'generator_version': 'v9-realtex', 'pattern': pattern, 'zones': zones, 'boundaries': bounds,
        'rmc_code': None, 'rock_rgb': zones[0]['rock_rgb'], 'joint_sets': zones[0]['joint_sets'],
        'shotcrete_rgb': [r.randint(120, 185)] * 3, 'floor_rgb': [r.randint(70, 100), r.randint(62, 85), r.randint(50, 70)],
        'face_grid_m': .025, 'fragments': r.randint(300, 900), 'muck_height_m': u(r, (.4, 2.)),
        'water': water, 'seepage': {'dry': 0, 'damp': 0, 'wet': r.randint(0, 3), 'dripping': r.randint(3, 8), 'flowing': r.randint(6, 14)}[water],
        'wet_level': {'dry': 0., 'damp': .25, 'wet': .5, 'dripping': .7, 'flowing': .9}[water],
        'half_barrels': r.random() < .5, 'barrel_spacing_m': u(r, (.4, .8)),
        'bolts': r.random() < .45, 'steel_sets': r.random() < .25, 'scale_bar': r.random() < .3, 'duct': r.random() < .4,
        'camera': {'location': [cx, -dist, cz], 'target': [tx, 0, tz], 'lens_mm': lens, 'key_light': key, 'closeup': closeup},
        'key_intensity': u(r, (.45, 1.05)), 'fill_intensity': u(r, (.0, .15)), 'flash': flash,
        'shadow_soft': u(r, (.02, .12)), 'spot_blend': u(r, (.15, .6)),
        # fine-texture contrast and crack darkness of the real-tile material, and the tile scale relative to output pixels
        'detail_strength': u(r, (1.6, 2.4)), 'crack_strength': u(r, (.4, .7)), 'tile_scale': u(r, (.8, 1.3)),
        # camera model applied by the labeller
        'camera_model': {'noise_sigma': u(r, (3., 10.)), 'jpeg_q': int(u(r, (55, 95))), 'blur': u(r, (.2, .6)),
                         'gamma': u(r, (.85, 1.2)), 'hotspot': u(r, (0., .25)) if flash else 0.},
        'light_kelvin': u(r, (4000, 6500)), 'white_balance': u(r, (.6, 1.)), 'exposure': u(r, (-.6, .4)),
        'capture_size': [1560, 1040],
    }
