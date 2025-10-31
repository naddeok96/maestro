#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-$PWD/data}"
MAKE_YOLO="${MAKE_YOLO:-true}"   # set to "false" to skip YOLO conversion

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  echo "[dry-run] Creating minimal test datasets"
  mkdir -p data/target/images/{train,val}
  if command -v convert >/dev/null 2>&1; then
    for split in train val; do
      for i in {1..10}; do
        convert -size 640x640 xc:gray "data/target/images/${split}/img_${i}.jpg"
      done
    done
  else
    python - <<'PY'
import base64
from pathlib import Path

GRAY_IMAGE = (
    """/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRof"""
    """Hh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAKAAoABAREA/8QAHwAAA"""
    """QUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBR"""
    """IhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdI"""
    """SUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztL"""
    """W2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/ACii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
)
try:
    from PIL import Image
except ImportError:
    Image = None

root = Path("data/target/images")
root.mkdir(parents=True, exist_ok=True)
img_bytes = base64.b64decode(GRAY_IMAGE)
for split in ("train", "val"):
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(1, 11):
        p = split_dir / f"img_{idx}.jpg"
        if Image is not None:
            Image.new("L", (640, 640), color=128).save(p, format="JPEG")
        else:
            p.write_bytes(img_bytes)
PY
  fi
  echo "[dry-run] (Note: no annotations; YOLO conversion will be skipped.)"
  exit 0
fi

echo "[*] Preparing COCO 2017 dataset"
mkdir -p coco && cd coco
if [[ ! -d images ]]; then
  mkdir -p images
  wget -c http://images.cocodataset.org/zips/train2017.zip
  wget -c http://images.cocodataset.org/zips/val2017.zip
  wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  unzip -n train2017.zip
  unzip -n val2017.zip
  unzip -n annotations_trainval2017.zip
  mkdir -p images
  mv train2017 images/train2017
  mv val2017 images/val2017
fi
cd ..

echo "[*] Preparing LVIS annotations"
mkdir -p lvis && cd lvis
[[ ! -f lvis_v1_train.json ]] && wget -c https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip
[[ ! -f lvis_v1_val.json   ]] && wget -c https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip
[[ -f lvis_v1_train.json.zip ]] && unzip -n lvis_v1_train.json.zip
[[ -f lvis_v1_val.json.zip   ]] && unzip -n lvis_v1_val.json.zip
cd ..

echo "[*] Preparing CrowdHuman (optional)"
mkdir -p crowdhuman && cd crowdhuman
if command -v gdown >/dev/null 2>&1; then
  gdown --no-cookies --fuzzy "https://drive.google.com/file/d/1G8wQfJNh8m9Z0l1Qe8O1YF7nK1kQfM4q/view" -O CrowdHuman_train.zip || true
  gdown --no-cookies --fuzzy "https://drive.google.com/file/d/1r8eYFzqTz0n7fQxY9pQdNqX0g3w9J1Vt/view" -O CrowdHuman_val.zip || true
  [[ -f CrowdHuman_train.zip ]] && unzip -n CrowdHuman_train.zip || true
  [[ -f CrowdHuman_val.zip   ]] && unzip -n CrowdHuman_val.zip || true
else
  echo "[!] gdown not found; skip automatic CrowdHuman download"
fi
cd ..

echo "[*] Target dataset placeholder"
mkdir -p target
echo "Place your COCO-format target dataset under: $DATA_DIR/target"
echo "  Expected structure (examples):"
echo "    target/images/train/*.jpg, target/annotations/instances_train.json"
echo "    target/images/val/*.jpg,   target/annotations/instances_val.json"

###############################################################################
# YOLO conversion (COCO/LVIS-style JSON → YOLO txt + dataset.yaml)
###############################################################################
if [[ "$MAKE_YOLO" == "true" ]]; then
  echo "[*] Converting datasets to YOLO format"

  python - <<'PY'
import json, yaml, shutil
from pathlib import Path

root = Path(".").resolve()

def coco_to_yolo(
    json_path: Path,
    images_dir: Path,
    labels_dir: Path,
    names_out: Path = None,
    skip_crowd: bool = True,
):
    if not json_path.exists():
        print(f"[skip] {json_path} not found")
        return None

    labels_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(json_path.read_text())
    # Build img dict
    img_by_id = {}
    for im in data.get("images", []):
        img_by_id[im["id"]] = {
            "file_name": im["file_name"],
            "width": im.get("width"),
            "height": im.get("height"),
        }

    # Build contiguous class map
    categories = data.get("categories", [])
    # Ensure deterministic ordering by ID then name
    categories = sorted(categories, key=lambda c: (c.get("id", 0), c.get("name", "")))
    catid_to_index = {c["id"]: i for i, c in enumerate(categories)}
    names = [c["name"] for c in categories]

    # Write names if requested
    if names_out is not None:
        names_out.parent.mkdir(parents=True, exist_ok=True)
        names_out.write_text("\n".join(names))

    # Collect anns per image
    anns_per_img = {}
    for ann in data.get("annotations", []):
        if skip_crowd and ann.get("iscrowd", 0) == 1:
            continue
        img_id = ann["image_id"]
        anns_per_img.setdefault(img_id, []).append(ann)

    # Convert
    num_boxes = 0
    for img_id, meta in img_by_id.items():
        fn = meta["file_name"]
        w, h = meta["width"], meta["height"]
        if not w or not h:
            # Try probing image if size missing
            try:
                from PIL import Image
                with Image.open(images_dir / fn) as im:
                    w, h = im.size
            except Exception:
                # Skip if size unknown
                continue

        label_lines = []
        for ann in anns_per_img.get(img_id, []):
            cat = catid_to_index.get(ann["category_id"])
            if cat is None:
                continue
            # COCO bbox = [x, y, w, h] in absolute pixels
            x, y, bw, bh = ann["bbox"]
            # YOLO needs normalized center x,y,w,h in [0,1]
            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h

            # Clip to [0,1] just in case
            def clip(v): return max(0.0, min(1.0, float(v)))
            xc, yc, nw, nh = map(clip, (xc, yc, nw, nh))

            label_lines.append(f"{cat} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            num_boxes += 1

        # Write label file next to image name (swap suffix to .txt)
        out_txt = labels_dir / Path(fn).with_suffix(".txt").name
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text("\n".join(label_lines))

    print(f"[ok] {json_path.name} → {labels_dir} ({num_boxes} boxes)")

    return names

def make_dataset_yaml(yaml_path: Path, train_images: Path, val_images: Path, names):
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    y = {
        "path": str(yaml_path.parent.resolve()),
        "train": str(train_images.resolve()),
        "val": str(val_images.resolve()),
        "names": names or [],
    }
    yaml_path.write_text(yaml.dump(y, sort_keys=False))
    print(f"[ok] wrote {yaml_path}")

# ------------------ COCO 2017 ------------------
coco_dir = root / "coco"
coco_img_train = coco_dir / "images/train2017"
coco_img_val   = coco_dir / "images/val2017"
coco_ann_train = coco_dir / "annotations/instances_train2017.json"
coco_ann_val   = coco_dir / "annotations/instances_val2017.json"
if coco_img_train.exists() and coco_ann_train.exists():
    coco_lbl_root = coco_dir / "labels"
    names = coco_to_yolo(coco_ann_train, coco_img_train, coco_lbl_root / "train2017", names_out=coco_dir / "names_coco.txt")
    _ = coco_to_yolo(coco_ann_val, coco_img_val, coco_lbl_root / "val2017")
    if names:
        make_dataset_yaml(coco_dir / "yolo_coco.yaml", coco_img_train, coco_img_val, names)

# ------------------ LVIS v1 (uses COCO image dirs) ------------------
lvis_dir = root / "lvis"
lvis_train = lvis_dir / "lvis_v1_train.json"
lvis_val   = lvis_dir / "lvis_v1_val.json"
# LVIS annotations typically reference COCO train/val2017 images
if lvis_train.exists():
    lvis_lbl_root = lvis_dir / "labels"
    names = coco_to_yolo(lvis_train, coco_img_train, lvis_lbl_root / "train2017", names_out=lvis_dir / "names_lvis.txt")
    if (lvis_val.exists()):
        _ = coco_to_yolo(lvis_val, coco_img_val, lvis_lbl_root / "val2017")
    if names:
        make_dataset_yaml(lvis_dir / "yolo_lvis.yaml", coco_img_train, coco_img_val, names)

# ------------------ Target dataset (COCO-style) ------------------
tgt = root / "target"
tgt_img_train = tgt / "images/train"
tgt_img_val   = tgt / "images/val"
tgt_ann_train = tgt / "annotations/instances_train.json"
tgt_ann_val   = tgt / "annotations/instances_val.json"
if tgt.exists() and tgt_ann_train.exists() and tgt_img_train.exists():
    tgt_lbl_root = tgt / "labels"
    names = coco_to_yolo(tgt_ann_train, tgt_img_train, tgt_lbl_root / "train", names_out=tgt / "names_target.txt")
    if tgt_ann_val.exists() and tgt_img_val.exists():
        _ = coco_to_yolo(tgt_ann_val, tgt_img_val, tgt_lbl_root / "val")
    # Prefer target images for dataset yaml
    if names:
        make_dataset_yaml(tgt / "yolo_target.yaml", tgt_img_train, tgt_img_val if tgt_img_val.exists() else tgt_img_train, names)

print("[✓] YOLO conversion complete")
PY

else
  echo "[i] MAKE_YOLO=false — skipping YOLO conversion"
fi

echo "[✓] Dataset preparation finished at $DATA_DIR"
