"""Tile label overlays of the synthetic tunnel faces into contact sheets for review."""
import argparse, json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', required=True, help='e.g. 301-320')
    ap.add_argument('--cols', type=int, default=4)
    ap.add_argument('--out', required=True)
    ap.add_argument('--data', default=str(ROOT / 'output'), help='output folder of batch_tunnel_faces.py')
    a = ap.parse_args()
    DATA = Path(a.data) / 'dataset'
    lo, hi = map(int, a.seeds.split('-'))
    coco = json.loads((DATA / '_annotations.coco.json').read_text())
    names = {c['id']: c['name'] for c in coco['categories']}
    by_file = {i['file_name']: i['id'] for i in coco['images']}
    tiles = []
    for s in range(lo, hi + 1):
        path = DATA / ('tunnel_face_%03d_overlay.jpg' % s)
        if not path.exists():
            continue
        img = cv2.imread(str(path))
        img = cv2.resize(img, (390, 260), interpolation=cv2.INTER_AREA)
        iid = by_file.get('tunnel_face_%03d.jpg' % s)
        codes = sorted({names[x['category_id']] for x in coco['annotations'] if x['image_id'] == iid} - {'Shortcrete00'})
        cv2.rectangle(img, (0, 236), (390, 260), (0, 0, 0), -1)
        cv2.putText(img, '%d  %s' % (s, ' + '.join(codes)), (6, 254), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        tiles.append(img)
    while len(tiles) % a.cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + a.cols]) for i in range(0, len(tiles), a.cols)]
    cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(a.out, len(rows), 'rows')


if __name__ == '__main__':
    main()
