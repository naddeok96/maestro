#!/usr/bin/env bash
set -euo pipefail

# ---------------------- Config toggles ----------------------
DATA_DIR="${1:-$PWD/data}"
MAKE_YOLO="${MAKE_YOLO:-true}"    # set to "false" to skip YOLO conversion
USE_VOC="${USE_VOC:-true}"        # set to "false" to skip PASCAL VOC download
# ------------------------------------------------------------

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
  mv -n train2017 images/train2017
  mv -n val2017 images/val2017
fi
cd ..

echo "[*] Preparing LVIS annotations"
mkdir -p lvis && cd lvis
[[ ! -f lvis_v1_train.json ]] && wget -c https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip
[[ ! -f lvis_v1_val.json   ]] && wget -c https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip
[[ -f lvis_v1_train.json.zip ]] && unzip -n lvis_v1_train.json.zip
[[ -f lvis_v1_val.json.zip   ]] && unzip -n lvis_v1_val.json.zip
cd ..

echo "[*] Target dataset placeholder"
mkdir -p target
echo "Place your COCO-format target dataset under: $DATA_DIR/target"
echo "  Expected structure (examples):"
echo "    target/images/train/*.jpg, target/annotations/instances_train.json"
echo "    target/images/val/*.jpg,   target/annotations/instances_val.json"

# ----------------------- Optional: PASCAL VOC -----------------------
echo "[*] Preparing PASCAL VOC 2007 + 2012 (optional)"
if [[ "$USE_VOC" == "true" ]]; then
  mkdir -p voc && cd voc

  # robust downloader with multiple mirrors + validation
  download_voc() {
    local outfile="$1"; shift
    local urls=("$@")

    # if already have a plausible file, validate and keep
    if [[ -f "$outfile" ]]; then
      if [[ $(stat -c%s "$outfile" 2>/dev/null || echo 0) -gt 10000000 ]] && tar -tf "$outfile" >/dev/null 2>&1; then
        echo "[ok] Using existing $outfile"
        return 0
      else
        echo "[!] $outfile exists but looks invalid; removing and redownloading"
        rm -f "$outfile"
      fi
    fi

    for u in "${urls[@]}"; do
      echo "[*] Trying $u"
      if [[ "$u" =~ ^https?://pjreddie\.com/ ]]; then
        # pjreddie sometimes returns HTML; use curl -L with a browser UA
        if curl -fL --retry 3 --retry-delay 2 \
             -A "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36" \
             -o "$outfile" "$u"; then
          :
        else
          echo "[!] Failed: $u"
          continue
        fi
      else
        if ! wget -c -O "$outfile" "$u"; then
          echo "[!] Failed: $u"
          continue
        fi
      fi

      # validate: size + tar list
      local sz
      sz=$(stat -c%s "$outfile" 2>/dev/null || echo 0)
      if [[ "$sz" -le 10000000 ]]; then
        echo "[!] Downloaded $outfile is too small ($sz bytes); likely HTML. Discarding."
        rm -f "$outfile"
        continue
      fi
      if ! tar -tf "$outfile" >/dev/null 2>&1; then
        echo "[!] $outfile failed tar validation; discarding."
        rm -f "$outfile"
        continue
      fi

      echo "[ok] Fetched valid $outfile"
      return 0
    done
    return 1
  }

  # Mirrors in priority order: DeepAI → pjreddie → Oxford
  download_voc VOCtrainval_06-Nov-2007.tar \
    https://data.deepai.org/VOCtrainval_06-Nov-2007.tar \
    https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar \
    http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar

  download_voc VOCtest_06-Nov-2007.tar \
    https://data.deepai.org/VOCtest_06-Nov-2007.tar \
    https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar \
    http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

  download_voc VOCtrainval_11-May-2012.tar \
    https://data.deepai.org/VOCtrainval_11-May-2012.tar \
    https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar \
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

  # Extract only after validation
  for t in VOCtrainval_06-Nov-2007.tar VOCtest_06-Nov-2007.tar VOCtrainval_11-May-2012.tar; do
    echo "[*] Extracting $t"
    tar -xf "$t"
  done

  cd ..
else
  echo "[i] USE_VOC=false — skipping VOC download"
fi


###############################################################################
# YOLO conversion (COCO/LVIS/VOC/target → YOLO)
###############################################################################
if [[ "$MAKE_YOLO" == "true" ]]; then
  echo "[*] Converting available datasets to YOLO format"

  python - <<'PY'
import json, yaml, os, shutil
from pathlib import Path

root = Path(".").resolve()

# ------------------ COCO/LVIS helper ------------------
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

    # Build image metadata by id
    img_by_id = {}
    for im in data.get("images", []):
        img_by_id[im["id"]] = {
            "file_name": im["file_name"],
            "width": im.get("width"),
            "height": im.get("height"),
        }

    # Contiguous categories (sorted by id then name for determinism)
    categories = sorted(data.get("categories", []), key=lambda c: (c.get("id", 0), c.get("name", "")))
    catid_to_index = {c["id"]: i for i, c in enumerate(categories)}
    names = [c["name"] for c in categories]

    if names_out is not None:
        names_out.parent.mkdir(parents=True, exist_ok=True)
        names_out.write_text("\n".join(names))

    # Index anns per image
    anns_per_img = {}
    for ann in data.get("annotations", []):
        if skip_crowd and ann.get("iscrowd", 0) == 1:
            continue
        anns_per_img.setdefault(ann["image_id"], []).append(ann)

    # Convert
    num_boxes = 0
    for img_id, meta in img_by_id.items():
        fn = meta["file_name"]
        w, h = meta.get("width"), meta.get("height")
        if not w or not h:
            try:
                from PIL import Image
                with Image.open(images_dir / fn) as im:
                    w, h = im.size
            except Exception:
                continue

        lines = []
        for ann in anns_per_img.get(img_id, []):
            cat = catid_to_index.get(ann["category_id"])
            if cat is None: 
                continue
            x, y, bw, bh = ann["bbox"]  # COCO xywh
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            def clip(v): return max(0.0, min(1.0, float(v)))
            cx, cy, nw, nh = map(clip, (cx, cy, nw, nh))
            lines.append(f"{cat} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            num_boxes += 1

        out_txt = labels_dir / Path(fn).with_suffix(".txt").name
        out_txt.write_text("\n".join(lines))

    print(f"[ok] {json_path.name} → {labels_dir} ({num_boxes} boxes)")
    return names

def write_dataset_yaml(yaml_path: Path, train_images: Path, val_images: Path, names):
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
        write_dataset_yaml(coco_dir / "yolo_coco.yaml", coco_img_train, coco_img_val, names)

# ------------------ LVIS v1 (on COCO images) ------------------
lvis_dir = root / "lvis"
lvis_train = lvis_dir / "lvis_v1_train.json"
lvis_val   = lvis_dir / "lvis_v1_val.json"
if lvis_train.exists():
    lvis_lbl_root = lvis_dir / "labels"
    names = coco_to_yolo(lvis_train, coco_img_train, lvis_lbl_root / "train2017", names_out=lvis_dir / "names_lvis.txt")
    if lvis_val.exists():
        _ = coco_to_yolo(lvis_val, coco_img_val, lvis_lbl_root / "val2017")
    if names:
        write_dataset_yaml(lvis_dir / "yolo_lvis.yaml", coco_img_train, coco_img_val, names)

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
    if names:
        write_dataset_yaml(tgt / "yolo_target.yaml", tgt_img_train, tgt_img_val if tgt_img_val.exists() else tgt_img_train, names)

# ------------------ PASCAL VOC → YOLO ------------------
voc_root = root / "voc"
if voc_root.exists() and (voc_root / "VOCdevkit").exists():
    from xml.etree import ElementTree as ET

    # VOC 20-class canonical order
    NAMES = [
        "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair",
        "cow","diningtable","dog","horse","motorbike","person","pottedplant",
        "sheep","sofa","train","tvmonitor"
    ]
    name_to_id = {n:i for i,n in enumerate(NAMES)}

    def ensure(p: Path): p.mkdir(parents=True, exist_ok=True)

    def voc_to_yolo(year_dir: Path, split: str, out_images: Path, out_labels: Path):
        jpeg = year_dir/"JPEGImages"
        ann  = year_dir/"Annotations"
        setfile = year_dir/"ImageSets/Main"/f"{split}.txt"
        if not setfile.exists():
            print(f"[skip] {setfile} not found")
            return 0,0
        ids = setfile.read_text().strip().splitlines()
        ensure(out_images); ensure(out_labels)

        imgs, boxes = 0, 0
        import os
        for img_id in ids:
            img_path = jpeg/f"{img_id}.jpg"
            if not img_path.exists():
                alt = jpeg/f"{img_id}.png"
                if alt.exists(): img_path = alt
                else: continue
            dest_img = out_images/img_path.name
            try:
                if not dest_img.exists():
                    os.symlink(img_path.resolve(), dest_img)
            except Exception:
                if not dest_img.exists():
                    shutil.copy2(img_path, dest_img)

            xml_path = ann/f"{img_id}.xml"
            if not xml_path.exists(): 
                continue
            tree = ET.parse(xml_path)
            w = int(tree.findtext("size/width"))
            h = int(tree.findtext("size/height"))

            lines = []
            for obj in tree.findall("object"):
                cls = obj.findtext("name")
                if cls not in name_to_id:
                    continue
                if obj.findtext("difficult") == "1":
                    continue
                bb = obj.find("bndbox")
                xmin = float(bb.findtext("xmin"))
                ymin = float(bb.findtext("ymin"))
                xmax = float(bb.findtext("xmax"))
                ymax = float(bb.findtext("ymax"))
                cx = ((xmin + xmax)/2.0) / w
                cy = ((ymin + ymax)/2.0) / h
                bw = (xmax - xmin) / w
                bh = (ymax - ymin) / h
                cid = name_to_id[cls]
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                boxes += 1

            (out_labels/img_path.with_suffix(".txt").name).write_text("\n".join(lines))
            imgs += 1
        return imgs, boxes

    img_train = voc_root/"images/train"
    img_val   = voc_root/"images/val"
    lbl_train = voc_root/"labels/train"
    lbl_val   = voc_root/"labels/val"
    ensure(img_train); ensure(img_val); ensure(lbl_train); ensure(lbl_val)

    splits = [
        ("VOC2007", "trainval"),
        ("VOC2007", "test"),
        ("VOC2012", "trainval"),
    ]

    total_imgs = total_boxes = 0
    for year, split in splits:
        year_dir = voc_root/"VOCdevkit"/year
        if not year_dir.exists(): 
            print(f"[skip] {year_dir} not found")
            continue
        if split in ("train","trainval"):
            imgs, boxes = voc_to_yolo(year_dir, split, img_train, lbl_train)
        else:
            imgs, boxes = voc_to_yolo(year_dir, split, img_val, lbl_val)
        total_imgs += imgs; total_boxes += boxes

    dataset_yaml = voc_root/"yolo_voc.yaml"
    y = {"path": str(voc_root), "train": str(img_train), "val": str(img_val), "names": NAMES}
    dataset_yaml.write_text(yaml.dump(y, sort_keys=False))
    print(f"[ok] VOC → YOLO in {voc_root} | images: {total_imgs}, boxes: {total_boxes}")
    print(f"[ok] wrote {dataset_yaml}")

print("[✓] YOLO conversion complete")
PY

else
  echo "[i] MAKE_YOLO=false — skipping YOLO conversion"
fi

echo "[✓] Dataset preparation finished at $DATA_DIR"

###############################################################################
# Small, public datasets (Clipart1k, Watercolor2k, Comic2k, PennFudan, KITTI)
###############################################################################

download_clipart1k() {
  local OUTDIR="clipart1k"
  mkdir -p "$OUTDIR"
  if [[ -n "${CLIPART1K_URL:-}" ]]; then
    echo "[*] Fetching Clipart1k from $CLIPART1K_URL"
    if wget -c -O "$OUTDIR/clipart1k.zip" "$CLIPART1K_URL"; then
      unzip -n "$OUTDIR/clipart1k.zip" -d "$OUTDIR"
    else
      echo "[!] Failed to download from CLIPART1K_URL"
    fi
  else
    echo "[i] Clipart1k needs a manual download (see project links)."
    echo "    Place VOC-format folders under: $OUTDIR/{JPEGImages,Annotations,ImageSets}"
    echo "    Reference: CVPR’18 Cross-Domain WSOD project page."
  fi
}

download_watercolor2k() {
  local OUTDIR="watercolor2k"
  mkdir -p "$OUTDIR"
  echo "[*] Watercolor2k page: https://opendatalab.com/OpenDataLab/Watercolor2k/download"
  echo "[i] Download the dataset and place VOC-style dirs into: $OUTDIR/{JPEGImages,Annotations,ImageSets}"
}

download_comic2k() {
  local OUTDIR="comic2k"
  mkdir -p "$OUTDIR"
  echo "[*] Comic2k page: https://opendatalab.com/OpenDataLab/Comic2k/download"
  echo "[i] Download and place VOC-style dirs into: $OUTDIR/{JPEGImages,Annotations,ImageSets}"
}

download_pennfudan() {
  local OUTDIR="pennfudan"
  mkdir -p "$OUTDIR"
  echo "[*] Downloading Penn-Fudan zip from UPenn"
  if wget -c -O "$OUTDIR/PennFudanPed.zip" "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"; then
    unzip -n "$OUTDIR/PennFudanPed.zip" -d "$OUTDIR"
  else
    echo "[!] Could not auto-download PennFudan. Download manually from the page and re-run."
  fi
}

download_kitti() {
  local OUTDIR="kitti"
  mkdir -p "$OUTDIR"
  echo "[*] KITTI official: https://www.cvlibs.net/datasets/kitti/"
  echo "[i] Please download 2D object detection 'data_object_image_2' and 'data_object_label_2' and place them under: $OUTDIR"
  echo "    $OUTDIR/image_2, $OUTDIR/label_2 (train split)"
}

write_yaml_clipart1k() {
  mkdir -p configs/datasets
  cat > configs/datasets/clipart1k.yaml <<'YAML'
path: ./data/clipart1k_yolo
train: images/train
val: images/val
names:
  0: aeroplane
  1: bicycle
  2: bird
  3: boat
  4: bottle
  5: bus
  6: car
  7: cat
  8: chair
  9: cow
  10: diningtable
  11: dog
  12: horse
  13: motorbike
  14: person
  15: pottedplant
  16: sheep
  17: sofa
  18: train
  19: tvmonitor
YAML
  echo "[ok] wrote configs/datasets/clipart1k.yaml"
}

write_yaml_watercolor2k() {
  mkdir -p configs/datasets
  cat > configs/datasets/watercolor2k.yaml <<'YAML'
path: ./data/watercolor2k_yolo
train: images/train
val: images/val
names:
  0: bicycle
  1: bird
  2: car
  3: cat
  4: dog
  5: person
YAML
  echo "[ok] wrote configs/datasets/watercolor2k.yaml"
}

write_yaml_comic2k() {
  mkdir -p configs/datasets
  cat > configs/datasets/comic2k.yaml <<'YAML'
path: ./data/comic2k_yolo
train: images/train
val: images/val
names:
  0: bicycle
  1: bird
  2: car
  3: cat
  4: dog
  5: person
YAML
  echo "[ok] wrote configs/datasets/comic2k.yaml"
}

write_yaml_pennfudan() {
  mkdir -p configs/datasets
  cat > configs/datasets/pennfudan.yaml <<'YAML'
path: ./data/pennfudan_yolo
train: images/train
val: images/val
names:
  0: person
YAML
  echo "[ok] wrote configs/datasets/pennfudan.yaml"
}

write_yaml_kitti() {
  mkdir -p configs/datasets
  cat > configs/datasets/kitti.yaml <<'YAML'
path: ./data/kitti_yolo
train: images/train
val: images/val
names:
  0: car
  1: pedestrian
  2: cyclist
YAML
  echo "[ok] wrote configs/datasets/kitti.yaml"
}

convert_voc_like_to_yolo() {
  local SRC_ROOT="$1"
  local OUT_ROOT="$2"
  local TRAIN_SPLIT="${3:-train}"
  local VAL_SPLIT="${4:-val}"

  SRC_ROOT="$SRC_ROOT" OUT_ROOT="$OUT_ROOT" TRAIN_SPLIT="$TRAIN_SPLIT" VAL_SPLIT="$VAL_SPLIT" python - <<'PY'
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

src_root = Path(os.environ["SRC_ROOT"]).resolve()
out_root = Path(os.environ["OUT_ROOT"]).resolve()
train_split = os.environ.get("TRAIN_SPLIT", "train")
val_split = os.environ.get("VAL_SPLIT", "val")

jpeg = src_root / "JPEGImages"
ann = src_root / "Annotations"
sets = src_root / "ImageSets" / "Main"

def read_ids(split_name: str) -> list[str]:
    txt = sets / f"{split_name}.txt"
    if txt.exists():
        return [line.strip() for line in txt.read_text().splitlines() if line.strip()]
    return [p.stem for p in jpeg.glob("*.jpg")]

def convert(ids: list[str], split: str) -> None:
    img_out = out_root / "images" / split
    lbl_out = out_root / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for sample_id in ids:
        img = jpeg / f"{sample_id}.jpg"
        if not img.exists():
            alt = jpeg / f"{sample_id}.png"
            if alt.exists():
                img = alt
            else:
                continue
        xml = ann / f"{sample_id}.xml"
        if not xml.exists():
            continue

        dst_img = img_out / img.name
        try:
            dst_img.symlink_to(img.resolve())
        except Exception:
            shutil.copy2(img, dst_img)

        root = ET.parse(xml).getroot()
        size = root.find("size")
        if size is None:
            (lbl_out / f"{sample_id}.txt").write_text("")
            continue
        try:
            width = float(size.findtext("width", default="0") or 0)
            height = float(size.findtext("height", default="0") or 0)
        except ValueError:
            width = height = 0.0
        if width <= 0 or height <= 0:
            (lbl_out / f"{sample_id}.txt").write_text("")
            continue

        lines = []
        for obj in root.findall("object"):
            name = obj.findtext("name", default="").strip()
            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            try:
                xmin = float(bbox.findtext("xmin", default="0"))
                ymin = float(bbox.findtext("ymin", default="0"))
                xmax = float(bbox.findtext("xmax", default="0"))
                ymax = float(bbox.findtext("ymax", default="0"))
            except ValueError:
                continue
            cx = ((xmin + xmax) / 2.0) / width
            cy = ((ymin + ymax) / 2.0) / height
            bw = max(0.0, xmax - xmin) / width
            bh = max(0.0, ymax - ymin) / height
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}  # {name}")

        (lbl_out / f"{sample_id}.txt").write_text("\n".join(lines))

train_ids = read_ids(train_split)
val_ids = read_ids(val_split)
convert(train_ids, "train")
convert(val_ids, "val")
print(f"[ok] VOC-like → YOLO at {out_root}")
PY
}

convert_pennfudan_to_yolo() {
  python - <<'PY'
import shutil
from pathlib import Path

src = Path("pennfudan") / "PennFudanPed"
if not src.exists():
    print("[!] PennFudan directory missing; skipping conversion")
    raise SystemExit(0)

out = Path("pennfudan_yolo")
(out / "images" / "train").mkdir(parents=True, exist_ok=True)
(out / "labels" / "train").mkdir(parents=True, exist_ok=True)

for img in (src / "PNGImages").glob("*.png"):
    dst = out / "images" / "train" / img.name
    try:
        dst.symlink_to(img.resolve())
    except Exception:
        shutil.copy2(img, dst)
    (out / "labels" / "train" / f"{img.stem}.txt").write_text("")

print("[i] PennFudan copied (labels placeholder; refine conversion as needed)")
PY
}

convert_kitti_to_yolo() {
  python - <<'PY'
import shutil
from pathlib import Path

src = Path("kitti")
img_dir = src / "image_2"
lbl_dir = src / "label_2"
if not img_dir.exists() or not lbl_dir.exists():
    print("[!] KITTI folders image_2/label_2 not found")
    raise SystemExit(0)

out = Path("kitti_yolo")
(out / "images" / "train").mkdir(parents=True, exist_ok=True)
(out / "labels" / "train").mkdir(parents=True, exist_ok=True)

for img in img_dir.glob("*.png"):
    dst = out / "images" / "train" / img.name
    try:
        dst.symlink_to(img.resolve())
    except Exception:
        shutil.copy2(img, dst)
    lbl_path = lbl_dir / f"{img.stem}.txt"
    lines = []
    if lbl_path.exists():
        for raw in lbl_path.read_text().splitlines():
            parts = raw.split()
            if len(parts) < 8:
                continue
            cls = parts[0].lower()
            if cls not in {"car", "pedestrian", "cyclist"}:
                continue
            try:
                left, top, right, bottom = map(float, parts[4:8])
            except ValueError:
                continue
            width = max(0.0, right - left)
            height = max(0.0, bottom - top)
            W, H = 1242.0, 375.0
            cx = (left + right) / (2.0 * W)
            cy = (top + bottom) / (2.0 * H)
            bw = width / W
            bh = height / H
            cid = {"car": 0, "pedestrian": 1, "cyclist": 2}[cls]
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    (out / "labels" / "train" / f"{img.stem}.txt").write_text("\n".join(lines))

print("[ok] KITTI → YOLO")
PY
}

prep_small_sets() {
  echo "[*] Preparing small public datasets"
  pushd "$DATA_DIR" >/dev/null

  download_clipart1k
  download_watercolor2k
  download_comic2k
  download_pennfudan
  download_kitti

  [[ -d clipart1k/JPEGImages ]] && convert_voc_like_to_yolo "clipart1k" "clipart1k_yolo" "train" "val"
  [[ -d watercolor2k/JPEGImages ]] && convert_voc_like_to_yolo "watercolor2k" "watercolor2k_yolo" "train" "test"
  [[ -d comic2k/JPEGImages ]] && convert_voc_like_to_yolo "comic2k" "comic2k_yolo" "train" "test"
  [[ -d pennfudan/PennFudanPed ]] && convert_pennfudan_to_yolo || true
  [[ -d kitti/image_2 ]] && convert_kitti_to_yolo || true

  popd >/dev/null

  write_yaml_clipart1k
  write_yaml_watercolor2k
  write_yaml_comic2k
  write_yaml_pennfudan
  write_yaml_kitti
}

# Trigger preparation after the core datasets are handled
prep_small_sets

