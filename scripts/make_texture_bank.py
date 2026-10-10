"""Build a class-blind bank of fine-texture tiles from rock regions of real training photographs.

Each tile is a square crop that lies fully inside one labelled rock region, converted to
grey and divided by its own low-frequency component, so it keeps only the detail finer
than about 20 px (crystal speckle, blast scars, dust, pits) at a mean of 1.0. The tiles
are stored as 8-bit PNGs with value = 128 * detail, and a JSON index records their source
image and class so leakage can be audited. Only the training split is used.

  python3 scripts/make_texture_bank.py --real "ROCK MASS LABELING.coco/train" --out textures --tiles 3000
"""
import argparse, json, random
from pathlib import Path
import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', required=True, help='folder with images and _annotations.coco.json (train split only)')
    ap.add_argument('--out', default='textures')
    ap.add_argument('--tiles', type=int, default=3000)
    ap.add_argument('--size', type=int, default=256, help='tile size in source pixels')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    real = Path(a.real); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d = json.loads((real / '_annotations.coco.json').read_text())
    names = {c['id']: c['name'] for c in d['categories']}
    images = {i['id']: i for i in d['images']}
    rock = [x for x in d['annotations'] if names[x['category_id']].startswith('RMC')]
    rnd = random.Random(a.seed); rnd.shuffle(rock)
    index, k = [], 0
    for ann in rock:
        if k >= a.tiles:
            break
        info = images[ann['image_id']]
        img = cv2.imread(str(real / info['file_name']))
        if img is None:
            continue
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        for p in ann['segmentation']:
            cv2.fillPoly(mask, [np.array(p).reshape(-1, 2).round().astype(np.int32)], 1)
        s = a.size
        inner = cv2.erode(mask, np.ones((s, s), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)      # centres whose s x s window lies inside the region
        ys, xs = np.nonzero(inner)
        if len(ys) == 0:
            continue
        for _ in range(2):                                    # up to two tiles per region
            j = rnd.randrange(len(ys)); y, x = ys[j] - s // 2, xs[j] - s // 2
            if y < 0 or x < 0 or y + s > h or x + s > w:
                continue
            crop = img[y:y + s, x:x + s]
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) + 1.
            low = cv2.GaussianBlur(g, (0, 0), 20)
            detail = np.clip(g / low, .35, 1.9)
            if detail.std() < .04 or g.mean() < 25:            # flat or black crops carry no texture
                continue
            name = 'tile_%05d.png' % k
            cv2.imwrite(str(out / name), np.clip(detail * 128, 0, 255).astype(np.uint8))
            index.append({'tile': name, 'image': info['file_name'], 'class': names[ann['category_id']],
                          'std': round(float(detail.std()), 4), 'mean_grey': round(float(g.mean()), 1)})
            k += 1
            if k >= a.tiles:
                break
    (out / 'index.json').write_text(json.dumps({'source_split': str(real), 'size': a.size, 'tiles': index}, indent=1))
    stds = [t['std'] for t in index]
    print('%d tiles from %d regions; detail std median %.3f (IQR %.3f-%.3f)' %
          (len(index), len({t['image'] for t in index}), np.median(stds), np.percentile(stds, 25), np.percentile(stds, 75)))


if __name__ == '__main__':
    main()
