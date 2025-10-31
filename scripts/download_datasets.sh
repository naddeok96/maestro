#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-$PWD/data}"
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
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"""
    """iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii="""
)

try:
    from PIL import Image
except ImportError:  # pragma: no cover - fallback for minimal environments
    Image = None

root = Path("data/target/images")
img_bytes = base64.b64decode(GRAY_IMAGE)
for split in ("train", "val"):
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(1, 11):
        out_path = split_dir / f"img_{idx}.jpg"
        if Image is not None:
            img = Image.new("L", (640, 640), color=128)
            img.save(out_path, format="JPEG")
        else:
            out_path.write_bytes(img_bytes)
PY
  fi
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
wget -c https://dl.fbaipublicfiles.com/lvis/lvis_v1_train.json.zip
wget -c https://dl.fbaipublicfiles.com/lvis/lvis_v1_val.json.zip
unzip -n lvis_v1_train.json.zip
unzip -n lvis_v1_val.json.zip
cd ..

echo "[*] Preparing CrowdHuman (optional)"
mkdir -p crowdhuman && cd crowdhuman
if command -v gdown >/dev/null 2>&1; then
  gdown --no-cookies --fuzzy "https://drive.google.com/file/d/1G8wQfJNh8m9Z0l1Qe8O1YF7nK1kQfM4q/view" -O CrowdHuman_train.zip || true
  gdown --no-cookies --fuzzy "https://drive.google.com/file/d/1r8eYFzqTz0n7fQxY9pQdNqX0g3w9J1Vt/view" -O CrowdHuman_val.zip || true
  [[ -f CrowdHuman_train.zip ]] && unzip -n CrowdHuman_train.zip || true
  [[ -f CrowdHuman_val.zip ]] && unzip -n CrowdHuman_val.zip || true
else
  echo "[!] gdown not found; skip automatic CrowdHuman download"
fi
cd ..

echo "[*] Target dataset placeholder"
mkdir -p target
echo "Place your COCO-format target dataset under: $DATA_DIR/target"

echo "[✓] Dataset preparation finished at $DATA_DIR"
