"""Turn a Rhino tunnel-face render into a labelled sample in the ROCK MASS LABELING schema.

Reads render/<name>_rgb.png and render/<name>_id.png written by rhino_tunnel_face.py.
The ID pass gives exact regions: exposed rock -> the job's RMC code, visible
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


def finish_photo(rgb, rng):
    """Downsample and add the flash fall-off, grain and JPEG look of a site photo."""
    img = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = ((xx - W * .52) / (W * .62)) ** 2 + ((yy - H * .45) / (H * .7)) ** 2
    img *= (1 - .38 * np.clip(r2, 0, 1.6))[..., None]
    img *= np.array([.96, 1.0, 1.05], np.float32)  # BGR: slight warm cast of tunnel lamps
    img = cv2.GaussianBlur(img, (0, 0), .6) + rng.normal(0, 3.5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default=None)
    ap.add_argument('--out', default=str(ROOT / 'output'))
    a = ap.parse_args()
    OUT = Path(a.out)
    job = json.loads((OUT / 'job.json').read_text())
    zone_codes = [z['code'] for z in job['zones']] if job.get('zones') else [job['rmc_code']]
    name = a.name or 'tunnel_face_%03d' % job['seed']
    categories = CATEGORIES
    cat = {c['name']: c['id'] for c in categories}
    for code in zone_codes:
        if code not in cat:
            raise SystemExit('rmc_code %s is not one of the dataset categories' % code)

    rgb = cv2.imread(str(OUT / 'render' / (name + '_rgb.png')))
    idm = cv2.imread(str(OUT / 'render' / (name + '_id.png')))
    if rgb is None or idm is None or rgb.shape != idm.shape:
        raise SystemExit('missing or mismatched RGB / ID captures')
    photo = finish_photo(rgb, np.random.default_rng(job['seed']))
    masks = region_masks(idm)

    img_dir = OUT / 'dataset/images'; lab_dir = OUT / 'dataset/labels'
    img_dir.mkdir(parents=True, exist_ok=True); lab_dir.mkdir(parents=True, exist_ok=True)
    file_name = name + '.jpg'
    cv2.imwrite(str(img_dir / file_name), photo, [cv2.IMWRITE_JPEG_QUALITY, 88])

    coco_path = OUT / 'dataset/_annotations.coco.json'
    coco = json.loads(coco_path.read_text()) if coco_path.exists() else {
        'info': {'description': 'Rhino synthetic tunnel faces, ROCK MASS LABELING schema',
                 'rmc_code_basis': 'assigned by the user per job; class codebook not defined in the source dataset'},
        'licenses': [{'id': 1, 'name': 'internal'}], 'categories': categories, 'images': [], 'annotations': []}
    old = [i['id'] for i in coco['images'] if i['file_name'] == file_name]
    coco['images'] = [i for i in coco['images'] if i['id'] not in old]
    coco['annotations'] = [x for x in coco['annotations'] if x['image_id'] not in old]
    image_id = max([i['id'] for i in coco['images']] + [-1]) + 1
    ann_id = max([x['id'] for x in coco['annotations']] + [-1]) + 1
    coco['images'].append({'id': image_id, 'license': 1, 'file_name': file_name, 'height': H, 'width': W,
                           'date_captured': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                           'extra': {'name': file_name, 'generator': 'rhino_tunnel_face.py', 'seed': job['seed']},
                           'source_dataset': 'rhino_synthetic', 'original_file_name': file_name})

    yolo, overlay = [], photo.copy()
    summary = {}
    # zones sharing a code are merged into one region, as a labeller would draw it
    regions = {}
    for zi, code in enumerate(zone_codes):
        regions[code] = regions.get(code, 0) | masks['rock%d' % zi]
    regions['Shortcrete00'] = masks['shotcrete']
    palette = [(60, 200, 60), (60, 60, 230), (220, 60, 220)]
    for n, (cname, mask) in enumerate(regions.items()):
        colour = (220, 120, 40) if cname == 'Shortcrete00' else palette[n % 3]
        polys = polygons(mask, min_area=.002 * W * H)
        if not polys:
            continue
        x, y, w, h = cv2.boundingRect(np.concatenate(polys).astype(np.int32))
        area = int(mask.sum())
        coco['annotations'].append({'id': ann_id, 'image_id': image_id, 'category_id': cat[cname],
                                    'bbox': [float(x), float(y), float(w), float(h)], 'iscrowd': 0,
                                    'area': float(area), 'segmentation': [p.astype(float).flatten().tolist() for p in polys]})
        ann_id += 1
        for p in polys:
            yolo.append(str(cat[cname]) + ' ' + ' '.join('%.8f' % v for v in (p / [W, H]).flatten()))
            fill = overlay.copy(); cv2.fillPoly(fill, [p.astype(np.int32)], colour)
            overlay = cv2.addWeighted(fill, .35, overlay, .65, 0)
            cv2.polylines(overlay, [p.astype(np.int32)], True, colour, 2)
        cv2.putText(overlay, cname, (x + 8, y + 24), cv2.FONT_HERSHEY_SIMPLEX, .7, colour, 2)
        summary[cname] = {'polygons': len(polys), 'pixels': area, 'fraction': round(area / (W * H), 3)}

    (lab_dir / (name + '.txt')).write_text('\n'.join(yolo) + '\n')
    coco_path.write_text(json.dumps(coco, indent=1))
    cv2.imwrite(str(OUT / 'dataset' / (name + '_overlay.jpg')), overlay)
    print(json.dumps({'image': str(img_dir / file_name), 'regions': summary}, indent=1))


if __name__ == '__main__':
    main()
