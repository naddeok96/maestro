#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-$PWD/data}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

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
