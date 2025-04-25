import os, sys, json, cv2, torch
import numpy as np
from collections import defaultdict

# 1 ── repo path -------------------------------------------------
REPO = r'D:\Pix2Vox'                 # ← change if needed
sys.path.append(REPO)

# 2 ── imports ---------------------------------------------------
from config import cfg
import utils.data_loaders
from utils.data_loaders import DatasetType

# 3 ── paths -----------------------------------------------------
RENDER_FMT = cfg.DATASETS.SHAPENET.RENDERING_PATH    # original PNG path
OUT_ROOT   = r'D:\Pix2Vox\results\images\test_input' # target folder
os.makedirs(OUT_ROOT, exist_ok=True)

# taxonomy_id → readable name
with open(cfg.DATASETS.SHAPENET.TAXONOMY_FILE_PATH, encoding='utf-8') as f:
    TAXNAME = {t['taxonomy_id']: t['taxonomy_name'] for t in json.load(f)}

# 4 ── build exactly the same test dataset (no transforms) -------
loader_cls = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET]
test_ds = loader_cls(cfg).get_dataset(
            DatasetType.TEST,
            cfg.CONST.N_VIEWS_RENDERING,
            transforms=None)

# iterator (batch_size = 1)
test_iter = iter(test_ds)

# 5 ── per-class counters ---------------------------------------
saved_per_class = defaultdict(int)
MAX_SAMPLES = 5        # save at most 5 samples per taxonomy
VIEWS       = 5        # copy 00.png … 04.png

for tax_id, samp_id, _, _ in test_iter:
    # tax_id may be tensor; convert to str
    tax_id = tax_id if isinstance(tax_id, str) else tax_id.item()

    # stop when every class has MAX_SAMPLES
    if all(c >= MAX_SAMPLES for c in saved_per_class.values()
           ) and len(saved_per_class) == len(TAXNAME):
        break

    if saved_per_class[tax_id] >= MAX_SAMPLES:
        continue

    cls_idx = saved_per_class[tax_id]          # 000, 001, …
    class_dir  = os.path.join(OUT_ROOT, TAXNAME[tax_id],
                              f'{cls_idx:03d}')
    os.makedirs(class_dir, exist_ok=True)

    for v in range(VIEWS):
        src = RENDER_FMT % (tax_id, samp_id, v)
        dst = os.path.join(class_dir, f'view{v}.png')
        cv2.imwrite(dst, cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB))

    saved_per_class[tax_id] += 1
    print(f'{TAXNAME[tax_id]}  sample {cls_idx:03d}  -> saved 5 views')

print('\nDone!  Raw inputs in:', OUT_ROOT)