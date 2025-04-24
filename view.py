from utils.data_loaders import ShapeNetDataLoader, DatasetType
from config import cfg
import os, shutil

# 1) Initialize loader exactly like runner.py
loader  = ShapeNetDataLoader(cfg)
dataset = loader.get_dataset(
    DatasetType.TEST,
    cfg.CONST.N_VIEWS_RENDERING,
    transforms=None
)

# 2) Create a folder to copy inputs into
dest_root = os.path.abspath('./output/images/test_input')
os.makedirs(dest_root, exist_ok=True)

# 3) For your first 3 samples, copy their input PNGs side-by-side
for sample_idx in range(3):
    print(f"\n--- Sample #{sample_idx} ---")
    taxonomy, sample_name, images, volume = dataset.get_datum(sample_idx)
    print("Taxonomy ID:", taxonomy)
    print("Sample name:", sample_name)
    for view_idx, img_path in enumerate(dataset.file_list[sample_idx]['rendering_images']):
        dst_dir = os.path.join(dest_root, f'{sample_idx:02d}')
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(img_path, dst_dir)
        print(f"  Copied view {view_idx} → {dst_dir}")