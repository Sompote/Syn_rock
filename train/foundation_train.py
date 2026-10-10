"""Synthetic pretraining of YOLO26s-seg and matched fine-tuning on the real ROCK MASS LABELING data.

Stage 1 (pretrain): COCO weights -> synthetic faces with GSI structure labels (7 classes).
Stage 2 (finetune): for each initialisation (coco, synthetic), real-train fraction and seed,
train with the exact protocol of experiments/rock_v1/PROTOCOL.txt (80 epochs, imgsz 640,
batch 16, AdamW, same augmentation) and evaluate with train_cloud.pixel_evaluate, the code
that produced the original baseline. Completed runs are skipped, so the script resumes.

  python3 train/foundation_train.py --syn /workspace/syn --real /workspace/rock --out /workspace/rock/output_foundation
"""
import argparse, csv, json, os, random, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--syn', required=True, help='folder holding dataset/images and dataset/labels_gsi')
ap.add_argument('--real', required=True, help='project root with dataset/ (prepared real data) and train_cloud.py')
ap.add_argument('--out', required=True)
ap.add_argument('--stage', default='all', choices=['pretrain', 'finetune', 'all', 'mix'])
ap.add_argument('--pretrain-epochs', type=int, default=40)
ap.add_argument('--pretrain-batch', type=int, default=32)
ap.add_argument('--fractions', default='1.0,0.1')
ap.add_argument('--small-seeds', default='42,43,44', help='seeds for fractions below 1')
ap.add_argument('--inits', default='coco,synthetic')
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--syn-weight', type=float, default=.25, help='loss (and so gradient) weight of synthetic batches; real = 1')
ap.add_argument('--mix-epochs', type=int, default=30)
ap.add_argument('--mix-fractions', default='0.1,1.0')
ap.add_argument('--syn-per-real', type=float, default=0.,
                help='if > 0, each epoch uses all real batches plus this many synthetic batches per real batch (domain-balanced)')
a = ap.parse_args()

REAL = Path(a.real); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
os.environ['ROCK_PROJECT_ROOT'] = str(REAL)
os.environ.setdefault('YOLO_CONFIG_DIR', str(REAL / 'yolo_config'))
sys.path.insert(0, str(REAL))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO, settings  # noqa: E402
import train_cloud as tc  # noqa: E402   original evaluation code
tc.OUT = OUT
settings.update({'wandb': False, 'mlflow': False, 'comet': False, 'clearml': False, 'neptune': False, 'dvc': False, 'hub': False})
COCO_WEIGHTS = str(REAL / 'yolo26s-seg.pt') if (REAL / 'yolo26s-seg.pt').exists() else 'yolo26s-seg.pt'
GSI_NAMES = ['intact_massive', 'blocky', 'very_blocky', 'blocky_disturbed_seamy', 'disintegrated', 'laminated_sheared', 'shotcrete']
PROTOCOL = dict(epochs=80, patience=0, imgsz=640, batch=16, device=0, seed=42, deterministic=True, amp=True, cache=False,
                optimizer='AdamW', lr0=.001, weight_decay=.0005, cos_lr=True, mosaic=0., mixup=0., copy_paste=0., degrees=0.,
                translate=.03, scale=.15, fliplr=0., flipud=0., hsv_h=0., hsv_s=.10, hsv_v=.15, overlap_mask=True, mask_ratio=2,
                save_period=-1, plots=True)


def log(*x):
    print(time.strftime('%H:%M:%S'), *x, flush=True)


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.symlink(src, dst)


def synthetic_layout():
    """YOLO layout over the synthetic set: 95 % train / 5 % val by a seeded shuffle, files symlinked."""
    root = Path(a.syn) / 'yolo_gsi'; ds = Path(a.syn) / 'dataset'
    stems = sorted(p.stem for p in (ds / 'labels_gsi').glob('*.txt') if (ds / 'images' / (p.stem + '.jpg')).exists())
    random.Random(0).shuffle(stems); nval = max(50, len(stems) // 20)
    for split, part in (('val', stems[:nval]), ('train', stems[nval:])):
        for s in part:
            link(ds / 'images' / (s + '.jpg'), root / 'images' / split / (s + '.jpg'))
            link(ds / 'labels_gsi' / (s + '.txt'), root / 'labels' / split / (s + '.txt'))
    (root / 'data.yaml').write_text('path: %s\ntrain: images/train\nval: images/val\nnames:\n%s' %
                                    (root, ''.join('  %d: %s\n' % (i, n) for i, n in enumerate(GSI_NAMES))))
    log('synthetic layout: %d train, %d val' % (len(stems) - nval, nval))
    return root / 'data.yaml'


def pretrain():
    best = OUT / 'pretrain_gsi/weights/best.pt'
    if best.exists():
        log('pretraining already done'); return best
    data = synthetic_layout()
    last = OUT / 'pretrain_gsi/weights/last.pt'
    model = YOLO(str(last) if last.exists() else COCO_WEIGHTS)
    model.train(resume=last.exists(), data=str(data), epochs=a.pretrain_epochs, patience=0, imgsz=640, batch=a.pretrain_batch,
                device=0, workers=a.workers, project=str(OUT), name='pretrain_gsi', exist_ok=True, seed=42, amp=True,
                optimizer='AdamW', lr0=.001, weight_decay=.0005, cos_lr=True, mosaic=0., mixup=0., copy_paste=0., degrees=0.,
                translate=.05, scale=.3, fliplr=.5, flipud=0., hsv_h=.01, hsv_s=.3, hsv_v=.3, overlap_mask=True, mask_ratio=2,
                save_period=5, plots=True)
    return best


def real_subset(frac, seed):
    """Prepared real data with only a seeded fraction of the training images; valid and test are complete."""
    full = REAL / 'dataset'
    if frac >= 1:
        return full / 'data.yaml'
    root = OUT / ('real_f%03d_s%d' % (round(frac * 100), seed))
    imgs = sorted((full / 'images/train').iterdir())
    pick = random.Random(seed).sample(imgs, max(1, round(frac * len(imgs))))
    for im in pick:
        link(im, root / 'images/train' / im.name)
        lab = full / 'labels/train' / (im.stem + '.txt')
        if lab.exists():
            link(lab, root / 'labels/train' / lab.name)
    for split in ('valid', 'test'):
        link(full / 'images' / split, root / 'images' / split); link(full / 'labels' / split, root / 'labels' / split)
    yaml = '\n'.join(('path: %s' % root) if l.startswith('path:') else l for l in (full / 'data.yaml').read_text().splitlines()) + '\n'
    (root / 'data.yaml').write_text(yaml)
    return root / 'data.yaml'


def finetune(init_weights):
    rows = []
    fracs = [float(f) for f in a.fractions.split(',')]
    for frac in fracs:
        seeds = [42] if frac >= 1 else [int(s) for s in a.small_seeds.split(',')]
        for seed in seeds:
            for init in a.inits.split(','):
                run = '%s_f%03d_s%d' % (init, round(frac * 100), seed)
                ev = OUT / run / 'evaluation.json'
                if not ev.exists():
                    weights = COCO_WEIGHTS if init == 'coco' else str(init_weights)
                    data = real_subset(frac, seed)
                    log('fine-tune', run, 'from', weights)
                    model = YOLO(weights)
                    model.train(data=str(data), project=str(OUT), name=run, exist_ok=True, workers=a.workers,
                                **dict(PROTOCOL, seed=seed))
                    del model; torch.cuda.empty_cache()
                    model = YOLO(str(OUT / run / 'weights/best.pt')); report = {}
                    for split in ('val', 'test'):
                        actual = 'valid' if split == 'val' else split
                        m = model.val(data=str(REAL / 'dataset/data.yaml'), split=split, imgsz=640, batch=16, device=0,
                                      workers=a.workers, project=str(OUT / run), name=split + '_evaluation', plots=False, verbose=False)
                        report[split] = {'instance_metrics': {k: float(v) for k, v in m.results_dict.items()},
                                         'pixel_metrics': tc.pixel_evaluate(model, actual, run)}
                    ev.write_text(json.dumps(report, indent=1)); del model; torch.cuda.empty_cache()
                r = json.loads(ev.read_text())['test']['pixel_metrics']
                rows.append({'run': run, 'init': init, 'fraction': frac, 'seed': seed, 'test_macro_iou_16': r['macro_iou_16'],
                             'ci_low': r['group_bootstrap_miou95ci'][0], 'ci_high': r['group_bootstrap_miou95ci'][1],
                             'test_macro_dice_16': r['macro_dice_16'], 'pixel_accuracy': r['pixel_accuracy']})
                log(run, 'test macro IoU %.4f' % r['macro_iou_16'])
                with open(OUT / 'finetune_summary.csv', 'w', newline='') as f:
                    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)


class DomainBatchSampler:
    """Batches drawn from one domain only (real or synthetic), reshuffled every epoch and interleaved at random.

    syn_per_real = 0 uses every synthetic image each epoch; syn_per_real = k > 0 draws k synthetic batches per
    real batch afresh each epoch, so the share of synthetic batches is fixed at k / (1 + k) whatever the pool size.
    """
    def __init__(self, is_syn, batch, seed, syn_per_real=0.):
        self.is_syn = np.asarray(is_syn); self.batch = batch; self.rng = np.random.default_rng(seed)
        self.real = np.flatnonzero(~self.is_syn); self.syn = np.flatnonzero(self.is_syn); self.k = syn_per_real

    def _n_real(self):
        return -(-len(self.real) // self.batch)

    def _n_syn(self):
        return int(round(self.k * self._n_real())) if self.k > 0 else -(-len(self.syn) // self.batch)

    def __iter__(self):
        real = self.rng.permutation(self.real)
        batches = [real[i:i + self.batch].tolist() for i in range(0, len(real), self.batch)]
        syn = self.rng.permutation(self.syn)[:self._n_syn() * self.batch] if self.k > 0 else self.rng.permutation(self.syn)
        batches += [syn[i:i + self.batch].tolist() for i in range(0, len(syn), self.batch)]
        for k in self.rng.permutation(len(batches)):
            yield batches[k]

    def __len__(self):
        return self._n_real() + self._n_syn()


def mix_trainer(syn_weight, syn_per_real=0.):
    """SegmentationTrainer on real + synthetic images, the loss of every synthetic batch scaled by syn_weight."""
    from ultralytics.models.yolo.segment.train import SegmentationTrainer
    from ultralytics.data.build import InfiniteDataLoader, seed_worker

    class MixTrainer(SegmentationTrainer):
        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
            if mode != 'train':
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            dataset = self.build_dataset(dataset_path, mode, batch_size)
            is_syn = [SYN_MARK in f for f in dataset.im_files]
            log('mixed train set: %d real, %d synthetic images, synthetic loss weight %.2f, synthetic batches per real batch %s'
                % (len(is_syn) - sum(is_syn), sum(is_syn), syn_weight, syn_per_real or 'all'))
            nw = min(os.cpu_count(), self.args.workers)
            return InfiniteDataLoader(dataset=dataset, batch_sampler=DomainBatchSampler(is_syn, batch_size, self.args.seed, syn_per_real),
                                      num_workers=nw, pin_memory=True, collate_fn=getattr(dataset, 'collate_fn', None),
                                      worker_init_fn=seed_worker, prefetch_factor=4 if nw > 0 else None)

        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            self._w = syn_weight if SYN_MARK in batch['im_file'][0] else 1.
            if not getattr(self, '_patched', False):        # scale the loss, and so the gradient, per batch
                orig = self.model.loss

                def weighted(b, preds=None):
                    loss, items = orig(b, preds)
                    return loss * self._w, items
                self.model.loss = weighted; self._patched = True
                log('loss weighting active: synthetic batches x %.2f, real x 1.00' % syn_weight)
            return batch
    return MixTrainer


SYN_MARK = os.sep + os.path.basename(os.path.normpath(a.syn)) + os.sep


def mixed(init='coco'):
    """Train on real (fraction) + all synthetic images with down-weighted synthetic loss; evaluate like finetune()."""
    rows = []
    syn_imgs = sorted(str(p) for p in (Path(a.syn) / 'dataset/images').glob('*.jpg') if (Path(a.syn) / 'dataset/labels' / (p.stem + '.txt')).exists())
    for frac in [float(f) for f in a.mix_fractions.split(',')]:
        seeds = [42] if frac >= 1 else [int(s) for s in a.small_seeds.split(',')]
        for seed in seeds:
            run = ('mixbal%g_w%03d_f%03d_s%d' % (a.syn_per_real, round(a.syn_weight * 100), round(frac * 100), seed) if a.syn_per_real > 0
                   else 'mix_w%03d_f%03d_s%d' % (round(a.syn_weight * 100), round(frac * 100), seed))
            ev = OUT / run / 'evaluation.json'
            if not ev.exists():
                data = real_subset(frac, seed)
                real_dir = Path(str(data).replace('data.yaml', '')) / 'images/train' if frac < 1 else REAL / 'dataset/images/train'
                lst = OUT / (run + '_train.txt')
                lst.write_text('\n'.join([str(p) for p in sorted(real_dir.iterdir())] + syn_imgs) + '\n')
                yaml = '\n'.join(('path: %s' % (REAL / 'dataset')) if l.startswith('path:') else ('train: %s' % lst) if l.startswith('train:') else l
                                  for l in (REAL / 'dataset/data.yaml').read_text().splitlines()) + '\n'
                (OUT / (run + '.yaml')).write_text(yaml)
                log('mixed', run)
                model = YOLO(COCO_WEIGHTS)
                model.train(data=str(OUT / (run + '.yaml')), project=str(OUT), name=run, exist_ok=True, workers=a.workers,
                            trainer=mix_trainer(a.syn_weight, a.syn_per_real), **dict(PROTOCOL, seed=seed, epochs=a.mix_epochs, close_mosaic=0))
                del model; torch.cuda.empty_cache()
                model = YOLO(str(OUT / run / 'weights/best.pt')); report = {}
                for split in ('val', 'test'):
                    actual = 'valid' if split == 'val' else split
                    m = model.val(data=str(REAL / 'dataset/data.yaml'), split=split, imgsz=640, batch=16, device=0,
                                  workers=a.workers, project=str(OUT / run), name=split + '_evaluation', plots=False, verbose=False)
                    report[split] = {'instance_metrics': {k: float(v) for k, v in m.results_dict.items()},
                                     'pixel_metrics': tc.pixel_evaluate(model, actual, run)}
                ev.write_text(json.dumps(report, indent=1)); del model; torch.cuda.empty_cache()
            r = json.loads(ev.read_text())['test']['pixel_metrics']
            log(run, 'test macro IoU %.4f' % r['macro_iou_16'])


assert torch.cuda.is_available(), 'GPU unavailable'
log('GPU', torch.cuda.get_device_name(0), 'torch', torch.__version__)
best = OUT / 'pretrain_gsi/weights/best.pt'
if a.stage in ('pretrain', 'all'):
    best = pretrain()
if a.stage in ('finetune', 'all'):
    finetune(best)
if a.stage == 'mix':
    mixed()
log('done')
