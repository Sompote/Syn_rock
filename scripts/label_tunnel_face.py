"""Turn a tunnel-face render (Rhino or Blender) into labelled samples: RMC labels and, for sampler jobs, GSI labels.

Reads render/<name>_rgb.png plus either render/<name>_index.npz (Blender object-index pass)
or render/<name>_id.png (Rhino flat-colour capture). Both give exact regions: exposed rock -> the job's RMC code, visible
shotcrete -> Shortcrete00. Muck, floor and background stay unlabelled, as in the
real data. Writes a 780 x 520 JPEG, COCO JSON with the original 16 categories,
a YOLO-seg label (class index = COCO category_id, as prepare_dataset.py does)
and an overlay for checking.
"""
import argparse, datetime, json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Categories of the ROCK MASS LABELING dataset, same ids and order.
CATEGORIES = [{"id": 0, "name": "RMCA01", "supercategory": "Face-F3"}, {"id": 1, "name": "RMCA02", "supercategory": "Face-F3"}, {"id": 2, "name": "RMCA03", "supercategory": "Face-F3"}, {"id": 3, "name": "RMCB01", "supercategory": "Face-F3"}, {"id": 4, "name": "RMCB02", "supercategory": "Face-F3"}, {"id": 5, "name": "RMCB03", "supercategory": "Face-F3"}, {"id": 6, "name": "RMCC01", "supercategory": "Face-F3"}, {"id": 7, "name": "RMCC02", "supercategory": "Face-F3"}, {"id": 8, "name": "RMCC03", "supercategory": "Face-F3"}, {"id": 9, "name": "RMCD01", "supercategory": "Face-F3"}, {"id": 10, "name": "RMCD02", "supercategory": "Face-F3"}, {"id": 11, "name": "RMCD03", "supercategory": "Face-F3"}, {"id": 12, "name": "RMCE01", "supercategory": "Face-F3"}, {"id": 13, "name": "RMCE02", "supercategory": "Face-F3"}, {"id": 14, "name": "RMCE03", "supercategory": "Face-F3"}, {"id": 15, "name": "Shortcrete00", "supercategory": "Face-F3"}]
W, H = 780, 520


def kelvin_bgr(k):
    """Approximate black-body colour at k kelvin (Tanner Helland fit) as BGR 0-1; matches blender/render_faces.py."""
    t = k / 100.
    r = 1. if t <= 66 else min(1., 1.292936 * (t - 60) ** -.1332047)
    g = min(1., max(0., (.39008158 * np.log(t) - .63184144) if t <= 66 else 1.12989086 * (t - 60) ** -.0755148))
    b = 1. if t >= 66 else (0. if t <= 19 else min(1., .54320679 * np.log(t - 10) - 1.19625409))
    return np.array([b, g, r], np.float32)


def finish_photo(rgb, rng, job=None):
    """Downsample and add the flash fall-off, camera white balance, grain and JPEG look of a site photo."""
    img = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = ((xx - W * .52) / (W * .62)) ** 2 + ((yy - H * .45) / (H * .7)) ** 2
    if job and job.get('light_kelvin'):
        # the camera corrects most of the lamp colour; 'white_balance' is how much (1 = fully neutral)
        c = kelvin_bgr(job['light_kelvin']); c = c / c.mean()
        img /= c[None, None, :] ** job.get('white_balance', .8)
        img *= (1 - .25 * np.clip(r2, 0, 1.6))[..., None]
        cm = job.get('camera_model')
        if cm:                                   # sensor and compression model: gamma, flash hot spot, noise, blur
            img = 255. * np.clip(img / 255., 0, 1) ** cm['gamma']
            if cm['hotspot'] > 0:
                hs = np.exp(-(((xx - W * .5) / (W * .35)) ** 2 + ((yy - H * .5) / (H * .35)) ** 2))
                img *= (1 + cm['hotspot'] * hs)[..., None]
            img = cv2.GaussianBlur(img, (0, 0), cm['blur']) + rng.normal(0, cm['noise_sigma'], img.shape)
            return np.clip(img, 0, 255).astype(np.uint8), cm['jpeg_q']
    else:
        img *= (1 - .38 * np.clip(r2, 0, 1.6))[..., None]
        img *= np.array([.96, 1.0, 1.05], np.float32)  # BGR: slight warm cast of tunnel lamps
    img = cv2.GaussianBlur(img, (0, 0), .6) + rng.normal(0, 3.5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), 88


# ID colours written by rhino_tunnel_face.py, in BGR: rock zones 0-2 and shotcrete.
ID_BGR = {'rock0': (0, 255, 0), 'rock1': (0, 0, 255), 'rock2': (255, 0, 255), 'shotcrete': (255, 0, 0)}


def region_masks(id_img):
    """Decode the unlit ID capture by chromaticity; mixed anti-aliased pixels stay unassigned."""
    px = id_img.astype(np.float32); peak = px.max(axis=-1)
    norm = px * 255 / np.maximum(peak, 1)[..., None]
    keys = list(ID_BGR)
    dist = np.stack([np.abs(norm - np.array(ID_BGR[k], np.float32)).max(axis=-1) for k in keys], -1)
    best = dist.argmin(-1); ok = (dist.min(-1) < 60) & (peak > 12)
    out = {}
    for n, key in enumerate(keys):
        m = ok & (best == n)
        small = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA) >= .5
        out[key] = small.astype(np.uint8)
    return out


def polygons(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        p = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2)
        if len(p) >= 3:
            polys.append(p)
    return polys


def masks_from_index(idx):
    """Exact class index from the Blender object-index pass: 1-3 rock zones, 4 shotcrete, 5 other, 0 background."""
    out = {}
    for key, k in (('rock0', 1), ('rock1', 2), ('rock2', 3), ('shotcrete', 4)):
        small = cv2.resize((idx == k).astype(np.float32), (W, H), interpolation=cv2.INTER_AREA) >= .5
        out[key] = small.astype(np.uint8)
    return out


# Pretraining label set: GSI structure classes (Hoek & Marinos) plus shotcrete; see docs/geological_patterns.md.
GSI_NAMES = ['intact_massive', 'blocky', 'very_blocky', 'blocky_disturbed_seamy', 'disintegrated', 'laminated_sheared', 'shotcrete']
GSI_CATEGORIES = [{'id': i, 'name': n, 'supercategory': 'GSI-structure'} for i, n in enumerate(GSI_NAMES)]
LABEL_SETS = {'rmc': ('_annotations.coco.json', 'labels', CATEGORIES),
              'gsi': ('_annotations_gsi.coco.json', 'labels_gsi', GSI_CATEGORIES)}


def annotate(regions, cat, overlay=None):
    """Regions {class name: mask} -> COCO annotations (no ids), YOLO lines and, optionally, a drawn overlay."""
    anns, yolo, summary = [], [], {}
    palette = [(60, 200, 60), (60, 60, 230), (220, 60, 220)]
    for n, (cname, mask) in enumerate(regions.items()):
        colour = (220, 120, 40) if cat[cname] == max(cat.values()) else palette[n % 3]
        polys = polygons(mask, min_area=.002 * W * H)
        if not polys:
            continue
        x, y, w, h = cv2.boundingRect(np.concatenate(polys).astype(np.int32))
        area = int(mask.sum())
        anns.append({'category_id': cat[cname], 'bbox': [float(x), float(y), float(w), float(h)], 'iscrowd': 0,
                     'area': float(area), 'segmentation': [p.astype(float).flatten().tolist() for p in polys]})
        for p in polys:
            yolo.append(str(cat[cname]) + ' ' + ' '.join('%.8f' % v for v in (p / [W, H]).flatten()))
            if overlay is not None:
                fill = overlay.copy(); cv2.fillPoly(fill, [p.astype(np.int32)], colour)
                overlay[:] = cv2.addWeighted(fill, .35, overlay, .65, 0)
                cv2.polylines(overlay, [p.astype(np.int32)], True, colour, 2)
        if overlay is not None:
            cv2.putText(overlay, cname, (x + 8, y + 24), cv2.FONT_HERSHEY_SIMPLEX, .7, colour, 2)
        summary[cname] = {'polygons': len(polys), 'pixels': area, 'fraction': round(area / (W * H), 3)}
    return anns, yolo, summary


def label_one(job, OUT):
    """Write the image, YOLO labels and overlay for one rendered job.

    Returns {'image': COCO image entry (no id), 'rmc': annotations, 'gsi': annotations or None, 'summary': ...}.
    Zones sharing a class are merged into one region, as a labeller would draw it.
    """
    OUT = Path(OUT)
    zones = job.get('zones') or [{'code': job['rmc_code']}]
    zone_codes = [z['code'] for z in zones]
    zone_gsi = [z.get('gsi') for z in zones]
    name = 'tunnel_face_%03d' % job['seed']
    cat = {c['name']: c['id'] for c in CATEGORIES}
    for code in zone_codes:
        if code not in cat:
            raise ValueError('rmc_code %s is not one of the dataset categories' % code)
    rgb = cv2.imread(str(OUT / 'render' / (name + '_rgb.png')))
    index = OUT / 'render' / (name + '_index.npz')
    if index.exists():                                   # Blender: exact index pass
        masks = masks_from_index(np.load(index)['index']); generator = 'blender/render_faces.py'
    else:                                                # Rhino: flat-colour ID capture
        idm = cv2.imread(str(OUT / 'render' / (name + '_id.png')))
        if idm is None or rgb is None or rgb.shape != idm.shape:
            raise ValueError('missing or mismatched RGB / ID captures for ' + name)
        masks = region_masks(idm); generator = 'rhino_tunnel_face.py'
    if rgb is None:
        raise ValueError('missing RGB render for ' + name)
    photo, jpeg_q = finish_photo(rgb, np.random.default_rng(job['seed']), job)
    img_dir = OUT / 'dataset/images'; img_dir.mkdir(parents=True, exist_ok=True)
    file_name = name + '.jpg'
    cv2.imwrite(str(img_dir / file_name), photo, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
    image = {'license': 1, 'file_name': file_name, 'height': H, 'width': W,
             'date_captured': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
             'extra': {'name': file_name, 'generator': generator, 'seed': job['seed'], 'pattern': job.get('pattern'),
                       'water': job.get('water'), 'closeup': job.get('camera', {}).get('closeup')},
             'source_dataset': 'synthetic', 'original_file_name': file_name}
    has_gsi = all(g is not None for g in zone_gsi)
    overlay = photo.copy()

    rmc_regions = {}
    for zi, code in enumerate(zone_codes):
        rmc_regions[code] = rmc_regions.get(code, 0) | masks['rock%d' % zi]
    rmc_regions['Shortcrete00'] = masks['shotcrete']
    rmc, yolo, summary = annotate(rmc_regions, cat, None if has_gsi else overlay)
    (OUT / 'dataset/labels').mkdir(parents=True, exist_ok=True)
    (OUT / 'dataset/labels' / (name + '.txt')).write_text('\n'.join(yolo) + '\n')

    gsi = None
    if has_gsi:
        gsi_regions = {}
        for zi, g in enumerate(zone_gsi):
            gsi_regions[GSI_NAMES[g]] = gsi_regions.get(GSI_NAMES[g], 0) | masks['rock%d' % zi]
        gsi_regions['shotcrete'] = masks['shotcrete']
        gsi, yolo_g, summary = annotate(gsi_regions, {n: i for i, n in enumerate(GSI_NAMES)}, overlay)
        (OUT / 'dataset/labels_gsi').mkdir(parents=True, exist_ok=True)
        (OUT / 'dataset/labels_gsi' / (name + '.txt')).write_text('\n'.join(yolo_g) + '\n')
    cv2.imwrite(str(OUT / 'dataset' / (name + '_overlay.jpg')), overlay)
    return {'image': image, 'rmc': rmc, 'gsi': gsi, 'summary': summary}


def merge_coco(OUT, labelled, label_set='rmc'):
    """Add (image, annotations) pairs to the label set's COCO file, replacing earlier entries of the same file."""
    fname, _, categories = LABEL_SETS[label_set]
    coco_path = Path(OUT) / 'dataset' / fname
    coco = json.loads(coco_path.read_text()) if coco_path.exists() else {
        'info': {'description': 'Synthetic tunnel faces, %s labels' % label_set,
                 'basis': ('RMC code assigned per job through an assumed mapping; no codebook in the source dataset'
                           if label_set == 'rmc' else 'GSI structure class of each zone, known exactly by the generator')},
        'licenses': [{'id': 1, 'name': 'internal'}], 'categories': categories, 'images': [], 'annotations': []}
    names = {image['file_name'] for image, _ in labelled}
    old = {i['id'] for i in coco['images'] if i['file_name'] in names}
    coco['images'] = [i for i in coco['images'] if i['id'] not in old]
    coco['annotations'] = [x for x in coco['annotations'] if x['image_id'] not in old]
    image_id = max([i['id'] for i in coco['images']] + [-1]) + 1
    ann_id = max([x['id'] for x in coco['annotations']] + [-1]) + 1
    for image, anns in labelled:
        coco['images'].append(dict(image, id=image_id))
        for x in anns:
            coco['annotations'].append(dict(x, id=ann_id, image_id=image_id)); ann_id += 1
        image_id += 1
    coco_path.parent.mkdir(parents=True, exist_ok=True)
    coco_path.write_text(json.dumps(coco))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'output'))
    a = ap.parse_args()
    job = json.loads((Path(a.out) / 'job.json').read_text())
    try:
        res = label_one(job, a.out)
    except ValueError as e:
        raise SystemExit(str(e))
    merge_coco(a.out, [(res['image'], res['rmc'])])
    if res['gsi'] is not None:
        merge_coco(a.out, [(res['image'], res['gsi'])], 'gsi')
    print(json.dumps({'image': str(Path(a.out) / 'dataset/images' / res['image']['file_name']), 'regions': res['summary']}, indent=1))


if __name__ == '__main__':
    main()
