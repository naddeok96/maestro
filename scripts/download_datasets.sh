#!/usr/bin/env bash
# Stable, strict, **idempotent** dataset fetcher + YOLO converter
# - Artistic datasets via gdown (Clipart/Watercolor/Comic) with auto-normalize
# - VOC via torchvision downloader
# - Skips work when artifacts already exist (safe to re-run)
#
# Usage:
#   bash scripts/download_datasets.sh [DATA_DIR]
#
# Env:
#   PYTHON_BIN=python3 VAL_FRACTION=0.2 DEBUG=1
#   # Optional direct URLs (tried before gdown IDs):
#   CLIPART1K_URL=..., WATERCOLOR2K_URL=..., COMIC2K_URL=...
#   # Optional custom Google Drive IDs (defaults set below):
#   CLIPART1K_GDRIVE_ID=..., WATERCOLOR2K_GDRIVE_ID=..., COMIC2K_GDRIVE_ID=...

set -Eeuo pipefail
shopt -s inherit_errexit

# ----------------- Global diagnostics -----------------
trap 'echo "[FATAL] ${BASH_SOURCE[0]}:$LINENO => $BASH_COMMAND" >&2' ERR
trap 'echo "[FATAL] interrupted (SIGINT/SIGTERM)" >&2; exit 2' INT TERM
: "${DEBUG:=0}"
if [[ "${DEBUG}" == "1" ]]; then
  export BASH_XTRACEFD=2
  export PS4='+ [${EPOCHREALTIME}] ${BASH_SOURCE##*/}:${LINENO}: '
  set -x
fi

# ----------------- Config -----------------
DATA_DIR_INPUT="${1:-$PWD/data}"
VAL_FRACTION="${VAL_FRACTION:-0.20}"   # autosplit ratio for train-only sets
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Default Google Drive IDs
: "${CLIPART1K_GDRIVE_ID:=1LvxwCOfUa-OklIvBJhB8zJlochjJiPFS}"
: "${WATERCOLOR2K_GDRIVE_ID:=1fa2L6oaPSjZ1_WqlTmIp6i2RbdR2y1Pw}"
: "${COMIC2K_GDRIVE_ID:=1bZtVWcxxFrijE_ALvNPjH1MXIKio6BIr}"

# ----------------- Utils -----------------
die() { echo "[FATAL] $*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }
log() { echo -e "$*"; }

# Merge directory A/* into B/ (rsync if present, else cp -a)
merge_dir() {
  local src="$1" dst="$2"
  if [[ -d "$src" ]]; then
    if command -v rsync >/dev/null 2>&1; then
      rsync -a "$src/" "$dst/" || cp -a "$src/." "$dst/"
    else
      mkdir -p "$dst"
      cp -a "$src/." "$dst/"
    fi
  fi
}

# GNU/BSD size helper
file_size() {
  local f="$1"
  if command -v stat >/dev/null 2>&1; then
    stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || wc -c <"$f"
  else
    wc -c <"$f"
  fi
}

is_valid_zip() { unzip -tq "$1" >/dev/null 2>&1; }
is_valid_tar() { tar -tf "$1" >/dev/null 2>&1; }

have_voc_triplet() {
  # Return 0 if CWD has JPEGImages/Annotations/ImageSets
  [[ -d JPEGImages && -d Annotations && -d ImageSets ]]
}

# ----------------- Resolve working dir (fail-loud) -----------------
DATA_DIR="$(
  cd "$(dirname "$DATA_DIR_INPUT")" \
  && mkdir -p "$(basename "$DATA_DIR_INPUT")" \
  && cd "$(basename "$DATA_DIR_INPUT")" \
  && pwd -P
)" || die "Failed to resolve DATA_DIR from: $DATA_DIR_INPUT"
[[ -w "$DATA_DIR" ]] || die "DATA_DIR not writable: $DATA_DIR"

cd "$DATA_DIR"
ROOT_DIR="$(pwd -P)"
log "[*] Using DATA_DIR: $ROOT_DIR"

# ----------------- Preconditions -----------------
need_cmd "$PYTHON_BIN"
need_cmd unzip
need_cmd tar

# ----------------- Downloader -----------------
fetch_file() {
  # fetch_file <outfile> <url1> [url2] ...
  local outfile="$1"; shift
  local urls=("$@")
  [[ "${#urls[@]}" -gt 0 ]] || die "fetch_file: no URLs provided for $outfile"

  if [[ -s "$outfile" ]]; then
    case "$outfile" in
      *.zip) if is_valid_zip "$outfile"; then log "[ok] Already valid: $outfile"; return 0; fi ;;
      *.tar|*.tar.gz|*.tgz) if is_valid_tar "$outfile"; then log "[ok] Already valid: $outfile"; return 0; fi ;;
      *) log "[ok] Already present: $outfile"; return 0 ;;
    esac
    log "[!] Existing $outfile invalid, refetching…"
    rm -f "$outfile"
  fi

  rm -f "$outfile".part 2>/dev/null || true
  for u in "${urls[@]}"; do
    [[ -z "$u" ]] && continue
    log "[*] Fetching $outfile from: $u"
    if command -v wget >/dev/null 2>&1; then
      if ! wget -c --tries=3 --timeout=60 -O "$outfile".part "$u"; then
        log "[!] wget failed: $u"; continue
      fi
    elif command -v curl >/dev/null 2>&1; then
      if ! curl -fL --retry 3 --retry-delay 2 --connect-timeout 20 --max-time 600 -o "$outfile".part "$u"; then
        log "[!] curl failed: $u"; continue
      fi
    else
      die "Neither wget nor curl found for download."
    fi

    local sz; sz=$(file_size "$outfile".part 2>/dev/null || echo 0)
    if [[ "$sz" -le 1024 ]]; then
      log "[!] Download too small ($sz bytes), trying next mirror"
      rm -f "$outfile".part
      continue
    fi

    local ok=false
    case "$outfile" in
      *.zip) is_valid_zip "$outfile".part && ok=true ;;
      *.tar|*.tar.gz|*.tgz) is_valid_tar "$outfile".part && ok=true ;;
      *) ok=true ;;
    esac

    if [[ "$ok" == "true" ]]; then
      mv "$outfile".part "$outfile"
      log "[ok] Valid: $outfile"
      return 0
    fi

    log "[!] Archive invalid; trying next mirror"
    rm -f "$outfile".part
  done

  die "All mirrors failed for $outfile"
}

# gdown helper (Drive ID)
gdown_fetch () {
  # gdown_fetch <outfile> <gdrive_file_id>
  local out="$1"; local fid="${2:-}"
  [[ -n "$fid" ]] || return 1
  if "$PYTHON_BIN" - "$fid" "$out".part <<'PY'
import sys
fid, out = sys.argv[1], sys.argv[2]
try:
    import gdown
except Exception:
    sys.exit(2)
url = f"https://drive.google.com/uc?id={fid}"
ok = gdown.download(url, out, quiet=False)
sys.exit(0 if ok else 1)
PY
  then
    mv "$out".part "$out"
    return 0
  else
    rm -f "$out".part >/dev/null 2>&1 || true
    return 1
  fi
}

extract_here() {
  local archive="$1"
  [[ -f "$archive" ]] || die "Archive not found: $archive"
  case "$archive" in
    *.zip) unzip -n "$archive" >/dev/null ;;
    *.tar) tar -xf "$archive" ;;
    *.tar.gz|*.tgz) tar -xzf "$archive" ;;
    *) die "Unknown archive format: $archive" ;;
  esac
}

# Find nested VOC root (dir containing JPEGImages+Annotations+ImageSets) and normalize into CWD
normalize_voc_here() {
  local found=""
  while IFS= read -r -d '' d; do
    if [[ -d "$d/JPEGImages" && -d "$d/Annotations" && -d "$d/ImageSets" ]]; then
      found="$d"; break
    fi
  done < <(find . -type d -print0)

  if [[ -z "$found" && -d ./clipart && -d ./clipart/JPEGImages && -d ./clipart/Annotations && -d ./clipart/ImageSets ]]; then
    found="./clipart"
  fi

  local base="$(basename "$(pwd -P)")"
  if [[ -z "$found" && -d "./$base" && -d "./$base/JPEGImages" && -d "./$base/Annotations" && -d "./$base/ImageSets" ]]; then
    found="./$base"
  fi

  if [[ -n "$found" && "$found" != "." ]]; then
    log "[*] Normalizing VOC layout from: $found -> $(pwd -P)"
    for sub in JPEGImages Annotations ImageSets; do
      if [[ -d "$sub" ]]; then
        merge_dir "$found/$sub" "$sub"
      else
        mv "$found/$sub" .
      fi
    done
  fi
}

# ----------------- Python deps (loud on failure) -----------------
log "[*] Checking Python deps (pillow, pyyaml, numpy, gdown)"
if ! "$PYTHON_BIN" - <<'PY'
import sys
try:
    import PIL, yaml, numpy  # noqa
    from PIL import Image    # noqa
    import gdown             # noqa
except Exception:
    sys.exit(1)
sys.exit(0)
PY
then
  log "[*] Installing pillow pyyaml numpy gdown …"
  "$PYTHON_BIN" -m pip install --upgrade --no-cache-dir pillow pyyaml numpy gdown
fi

# ----------------- Small sets (artistic via gdown, with auto-normalize) -----------------
log "[*] Preparing artistic datasets via Google Drive (Clipart1k, Watercolor2k, Comic2k)"

# Helper to fetch (URL override first, then gdown ID), then unzip & auto-normalize VOC layout
fetch_unzip_voc() {
  # fetch_unzip_voc <folder_name> <outfile.zip> <URL_env> <GDRIVE_ID>
  local folder="$1"; local zipname="$2"; local url_env="$3"; local gid="$4"
  mkdir -p "$folder"
  pushd "$folder" >/dev/null

  if have_voc_triplet; then
    log "[ok] $folder already prepared, skipping download/extract"
    popd >/dev/null; return 0
  fi

  if [[ -n "$url_env" ]]; then
    fetch_file "$zipname" "$url_env"
  else
    gdown_fetch "$zipname" "$gid" || die "Failed to download $folder via gdown"
  fi
  extract_here "$zipname" || true
  rm -f "$zipname"

  normalize_voc_here
  have_voc_triplet || die "$folder structure missing {JPEGImages,Annotations,ImageSets}"
  popd >/dev/null
  log "[ok] $folder ready"
}

# Clipart1k, Watercolor2k, Comic2k
fetch_unzip_voc "clipart1k"    "clipart.zip"    "${CLIPART1K_URL:-}"    "$CLIPART1K_GDRIVE_ID"
fetch_unzip_voc "watercolor2k" "watercolor.zip" "${WATERCOLOR2K_URL:-}" "$WATERCOLOR2K_GDRIVE_ID"
fetch_unzip_voc "comic2k"      "comic.zip"      "${COMIC2K_URL:-}"      "$COMIC2K_GDRIVE_ID"

# ----------------- Penn-Fudan (mask dataset) -----------------
mkdir -p pennfudan
pushd pennfudan >/dev/null
if [[ -d PennFudanPed/PNGImages && -d PennFudanPed/PedMasks ]]; then
  log "[ok] PennFudan already extracted, skipping"
else
  fetch_file PennFudanPed.zip "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
  extract_here PennFudanPed.zip
  [[ -d PennFudanPed/PNGImages && -d PennFudanPed/PedMasks ]] || die "PennFudan structure unexpected"
fi
popd >/dev/null

# ----------------- KITTI 2D: images + labels -----------------
mkdir -p kitti_raw
pushd kitti_raw >/dev/null
if [[ -d training/image_2 && -d training/label_2 ]]; then
  log "[ok] KITTI raw already extracted, skipping"
else
  fetch_file data_object_image_2.zip "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
  fetch_file data_object_label_2.zip "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"
  extract_here data_object_image_2.zip
  extract_here data_object_label_2.zip
fi
popd >/dev/null

mkdir -p kitti
if [[ -d kitti_raw/training/image_2 && -d kitti_raw/training/label_2 ]]; then
  ln -sfn "$ROOT_DIR/kitti_raw/training/image_2" "$ROOT_DIR/kitti/image_2"
  ln -sfn "$ROOT_DIR/kitti_raw/training/label_2" "$ROOT_DIR/kitti/label_2"
else
  die "KITTI training folders not found after extraction"
fi

# ----------------- COCO 2017 -----------------
log "[*] Preparing COCO 2017"
mkdir -p coco
pushd coco >/dev/null
if [[ -d images/train2017 && -d images/val2017 && -f annotations/instances_train2017.json ]]; then
  log "[ok] COCO2017 already present, skipping download"
else
  mkdir -p images annotations
  fetch_file train2017.zip "http://images.cocodataset.org/zips/train2017.zip"
  fetch_file val2017.zip   "http://images.cocodataset.org/zips/val2017.zip"
  fetch_file annotations_trainval2017.zip "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
  extract_here train2017.zip
  extract_here val2017.zip
  extract_here annotations_trainval2017.zip
  mv -n train2017 images/train2017 || true
  mv -n val2017   images/val2017   || true
fi
popd >/dev/null

# ----------------- LVIS v1 (annotations only) -----------------
log "[*] Preparing LVIS v1 (annotations only)"
mkdir -p lvis
pushd lvis >/dev/null
if [[ -f lvis_v1_train.json && -f lvis_v1_val.json ]]; then
  log "[ok] LVIS jsons already present, skipping"
else
  fetch_file lvis_v1_train.json.zip "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip"
  fetch_file lvis_v1_val.json.zip   "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip"
  extract_here lvis_v1_train.json.zip
  extract_here lvis_v1_val.json.zip
  [[ -f lvis_v1_train.json && -f lvis_v1_val.json ]] || die "LVIS jsons missing after unzip"
fi
popd >/dev/null



# ---- COCO compatibility links for summary (generic train/val) ----
mkdir -p "$ROOT_DIR/coco/images" "$ROOT_DIR/coco/labels"
ln -sfn "$ROOT_DIR/coco/images/train2017" "$ROOT_DIR/coco/images/train"
ln -sfn "$ROOT_DIR/coco/images/val2017"   "$ROOT_DIR/coco/images/val"
ln -sfn "$ROOT_DIR/coco/labels/train2017" "$ROOT_DIR/coco/labels/train" || true
ln -sfn "$ROOT_DIR/coco/labels/val2017"   "$ROOT_DIR/coco/labels/val"   || true

# ---- LVIS labels: create direct train/val dirs (no symlinks for consistency) ----
mkdir -p "$ROOT_DIR/lvis/labels/train" "$ROOT_DIR/lvis/labels/val" "$ROOT_DIR/lvis/images"

# ---- Ensure LVIS image dirs are *real directories* (not symlinks), then materialize subsets ----
for d in "$ROOT_DIR/lvis/images/train" "$ROOT_DIR/lvis/images/val"; do
  if [[ -L "$d" ]]; then rm -f "$d"; fi
  mkdir -p "$d"
done

log "[*] Linking LVIS image subsets from COCO"
"$PYTHON_BIN" - <<'PY'
import json, os, sys
from pathlib import Path

ROOT = Path(".").resolve()
coco_train = ROOT/"coco/images/train2017"
coco_val   = ROOT/"coco/images/val2017"
lvis_dir   = ROOT/"lvis"
out_train  = lvis_dir/"images/train"
out_val    = lvis_dir/"images/val"

def ensure(p: Path): p.mkdir(parents=True, exist_ok=True)

def link_subset(lvis_json: Path, out_dir: Path):
    ensure(out_dir)
    if not lvis_json.exists():
        print(f"[skip] {lvis_json} missing")
        return 0
    
    print(f"[*] Processing {lvis_json.name}...")
    data = json.loads(lvis_json.read_text())
    imgs = data.get("images", [])
    ok = 0
    failed = 0
    
    for im in imgs:
        # LVIS uses coco_url instead of file_name
        fn = im.get("file_name")
        if not fn:
            coco_url = im.get("coco_url", "")
            if coco_url:
                fn = Path(coco_url).name
            else:
                # Fallback: use id
                img_id = im.get("id")
                if img_id:
                    fn = f"{int(img_id):012d}.jpg"
                else:
                    failed += 1
                    continue
        
        dst = out_dir / fn
        if dst.exists():
            ok += 1
            continue
        
        # Find source in COCO
        src = None
        for base in (coco_train, coco_val):
            cand = base / fn
            if cand.exists():
                src = cand
                break
        
        if src is None:
            # Try basename fallback
            basefn = Path(fn).name
            for base in (coco_train, coco_val):
                cand = base / basefn
                if cand.exists():
                    src = cand
                    break
        
        if src is None:
            failed += 1
            continue
        
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(src.resolve(), dst)
            ok += 1
        except Exception:
            import shutil
            try:
                shutil.copy2(src, dst)
                ok += 1
            except Exception:
                failed += 1
    
    print(f"[ok] LVIS subset -> {out_dir.name}: {ok} images linked")
    if failed > 0:
        print(f"[!] Failed to link {failed} images")
    return ok

n_tr = link_subset(lvis_dir/"lvis_v1_train.json", out_train)
n_va = link_subset(lvis_dir/"lvis_v1_val.json",   out_val)

if n_tr == 0 and n_va == 0:
    print("[FATAL] No LVIS images were linked!", file=sys.stderr)
    sys.exit(1)
PY

mkdir -p "$ROOT_DIR/lvis/labels"

# link or copy labels/train2017 -> labels/train
if [[ -d "$ROOT_DIR/lvis/labels/train2017" ]]; then
  ln -sfn "$ROOT_DIR/lvis/labels/train2017" "$ROOT_DIR/lvis/labels/train" 2>/dev/null \
    || { rm -rf "$ROOT_DIR/lvis/labels/train"; cp -a "$ROOT_DIR/lvis/labels/train2017" "$ROOT_DIR/lvis/labels/train"; }
fi

# link or copy labels/val2017 -> labels/val
if [[ -d "$ROOT_DIR/lvis/labels/val2017" ]]; then
  ln -sfn "$ROOT_DIR/lvis/labels/val2017" "$ROOT_DIR/lvis/labels/val" 2>/dev/null \
    || { rm -rf "$ROOT_DIR/lvis/labels/val"; cp -a "$ROOT_DIR/lvis/labels/val2017" "$ROOT_DIR/lvis/labels/val"; }
fi


# ----------------- PASCAL VOC 2007+2012 via Hugging Face (HF-only, robust normalize) -----------------
log "[*] Preparing PASCAL VOC 2007+2012"

mkdir -p voc
pushd voc >/dev/null

if [[ -d VOCdevkit/VOC2007 && -d VOCdevkit/VOC2012 ]]; then
  log "[ok] VOCdevkit already present, skipping download"
else
  : "${PYTHON_BIN:=${PYTHON_BIN:-python3}}"
  : "${VOC_HF_REPO:=HuggingFaceM4/pascal_voc}"   # dataset repo with the original tarballs
  : "${VOC_HF_REVISION:=main}"
  : "${VOC_2007_ARCHIVE:=voc2007.tar.gz}"        # 2007 trainval + test
  : "${VOC_2012_ARCHIVE:=voc2012.tar.gz}"        # 2012 trainval

  need_cmd tar

  hf_download() {
    # hf_download <repo_id> <filename> <revision> <outpath>
    local repo="$1" fn="$2" rev="$3" out="$4"
    "$PYTHON_BIN" - "$repo" "$fn" "$rev" "$out" <<'PY'
import sys, os, shutil
repo, fn, rev, out = sys.argv[1:5]
try:
    from huggingface_hub import hf_hub_download
except Exception:
    sys.exit(2)  # library missing
try:
    path = hf_hub_download(repo_id=repo, filename=fn, revision=rev, repo_type="dataset")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if os.path.abspath(path) != os.path.abspath(out):
        shutil.copy2(path, out)
    print("[ok] hf_hub:", fn)
    sys.exit(0)
except Exception as e:
    print("[!] hf_hub failed:", e)
    sys.exit(1)
PY
    return $?
  }

  # Ensure huggingface_hub is installed
  if ! "$PYTHON_BIN" - <<'PY'
try:
    import huggingface_hub  # noqa
except Exception:
    raise SystemExit(1)
PY
  then
    log "[*] Installing huggingface_hub …"
    "$PYTHON_BIN" -m pip install --upgrade --no-cache-dir huggingface_hub || die "Failed to install huggingface_hub"
  fi

  # Download archives if missing/invalid
  for fn in "$VOC_2007_ARCHIVE" "$VOC_2012_ARCHIVE"; do
    [[ -z "$fn" ]] && continue
    if [[ -s "$fn" ]] && tar -tf "$fn" >/dev/null 2>&1; then
      log "[ok] Already valid: $fn"
      continue
    fi
    rm -f "$fn"
    log "[*] Fetching from HF (dataset): $VOC_HF_REPO@$VOC_HF_REVISION :: $fn"
    hf_download "$VOC_HF_REPO" "$fn" "$VOC_HF_REVISION" "$fn" || die "HF download failed for $fn (repo: $VOC_HF_REPO)"
    tar -tf "$fn" >/dev/null 2>&1 || die "Corrupt archive after download: $fn"
  done

  # Extract (don't assume exact paths inside the tars)
  log "[*] Extracting VOC tarballs (HF)"
  case "$VOC_2007_ARCHIVE" in *.tar.gz|*.tgz) tar -xzf "$VOC_2007_ARCHIVE";; *.tar) tar -xf "$VOC_2007_ARCHIVE";; *) tar -xf "$VOC_2007_ARCHIVE";; esac
  case "$VOC_2012_ARCHIVE" in *.tar.gz|*.tgz) tar -xzf "$VOC_2012_ARCHIVE";; *.tar) tar -xf "$VOC_2012_ARCHIVE";; *) tar -xf "$VOC_2012_ARCHIVE";; esac

  # --- Normalize to VOCdevkit/{VOC2007,VOC2012} regardless of how the archives unpacked ---
  normalize_voc_tree() {
    mkdir -p VOCdevkit
    # 1) If a nested VOCdevkit already exists somewhere, merge it in
    while IFS= read -r -d '' nested_voc; do
      for yr in VOC2007 VOC2012; do
        if [[ -d "$nested_voc/$yr" ]]; then
          log "[*] Merging nested $nested_voc/$yr -> VOCdevkit/$yr"
          mkdir -p "VOCdevkit/$yr"
          merge_dir "$nested_voc/$yr/JPEGImages" "VOCdevkit/$yr/JPEGImages"
          merge_dir "$nested_voc/$yr/Annotations" "VOCdevkit/$yr/Annotations"
          merge_dir "$nested_voc/$yr/ImageSets"  "VOCdevkit/$yr/ImageSets"
        fi
      done
    done < <(find . -type d -name VOCdevkit -print0)

    # 2) Any directory that *looks* like a VOC root (has the triplet) — classify by year and merge
    while IFS= read -r -d '' d; do
      # Skip our final destination
      [[ "$d" == "./VOCdevkit/VOC2007" || "$d" == "./VOCdevkit/VOC2012" ]] && continue
      if [[ -d "$d/JPEGImages" && -d "$d/Annotations" && -d "$d/ImageSets" ]]; then
        local year=""
        case "$d" in
          *2007*|*VOC2007*) year=2007 ;;
          *2012*|*VOC2012*) year=2012 ;;
          *) # Heuristic: if it has test split -> likely 2007; else assume 2012
             if [[ -f "$d/ImageSets/Main/test.txt" ]]; then year=2007; else year=2012; fi ;;
        esac
        log "[*] Normalizing $(realpath --relative-to=. "$d") → VOCdevkit/VOC${year}"
        mkdir -p "VOCdevkit/VOC${year}"
        merge_dir "$d/JPEGImages" "VOCdevkit/VOC${year}/JPEGImages"
        merge_dir "$d/Annotations" "VOCdevkit/VOC${year}/Annotations"
        merge_dir "$d/ImageSets"  "VOCdevkit/VOC${year}/ImageSets"
      fi
    done < <(find . -maxdepth 5 -type d -print0)
  }

  normalize_voc_tree

  # Sanity check
  [[ -d VOCdevkit/VOC2007 && -d VOCdevkit/VOC2012 ]] || die "VOCdevkit not properly prepared (expected VOC2007 & VOC2012)"
fi

popd >/dev/null

# ----------------- YOLO conversions (idempotent) -----------------
log "[*] Converting all datasets to YOLO (skip-aware)"

VAL_FRACTION="$VAL_FRACTION" "$PYTHON_BIN" - <<'PY'
import json, yaml, os, shutil, random, sys
from pathlib import Path
from xml.etree import ElementTree as ET

try:
    from PIL import Image
    import numpy as np
except Exception as e:
    print("[FATAL] Missing python deps:", e, file=sys.stderr); sys.exit(1)

ROOT = Path(".").resolve()
try:
    VAL_FRACTION = float(os.environ.get("VAL_FRACTION","0.2"))
    if not (0.0 < VAL_FRACTION < 1.0): raise ValueError
except Exception:
    print("[FATAL] VAL_FRACTION must be in (0,1)", file=sys.stderr); sys.exit(1)

def ensure(p: Path): p.mkdir(parents=True, exist_ok=True)
def nonempty(p: Path): return p.exists() and any(p.iterdir())

def write_yaml_abs(yaml_path: Path, ds_root: Path, train_images: Path, val_images: Path, names):
    ds_root = ds_root.resolve()
    y = {"path": str(ds_root), "train": str(train_images.resolve()), "val": str(val_images.resolve()), "names": list(names or [])}
    ensure(yaml_path.parent)
    if yaml_path.exists():
        print(f"[ok] {yaml_path} exists, leaving as-is")
    else:
        yaml_path.write_text(yaml.dump(y, sort_keys=False))
        print(f"[ok] wrote {yaml_path}")

def coco_to_yolo(json_path: Path, images_dir: Path, labels_dir: Path, names_out: Path=None, skip_crowd=True):
    if not json_path.exists():
        print(f"[skip] {json_path} missing"); return None
    ensure(labels_dir)
    # If labels already look populated, skip heavy convert
    if any(labels_dir.glob("*.txt")):
        print(f"[ok] labels exist for {json_path.name}, skipping conversion")
        # Still return names for YAML
        try:
            data = json.loads(json_path.read_text())
            categories = sorted(data.get("categories", []), key=lambda c: (c.get("id",0), c.get("name","")))
            names = [c["name"] for c in categories]
            if names_out is not None and not names_out.exists():
                ensure(names_out.parent); names_out.write_text("\n".join(names))
            return names
        except Exception:
            return None

    data = json.loads(json_path.read_text())
    img_by_id = {}
    for im in data.get("images", []):
        fn = im.get("file_name")
        if not fn:
            cu = im.get("coco_url","")
            fn = Path(cu).name if cu else f"{int(im['id']):012d}.jpg"
        img_by_id[im["id"]] = {"file_name": fn, "width": im.get("width"), "height": im.get("height")}
    categories = sorted(data.get("categories", []), key=lambda c: (c.get("id",0), c.get("name","")))
    catid_to_index = {c["id"]: i for i,c in enumerate(categories)}
    names = [c["name"] for c in categories]
    if names_out is not None and not names_out.exists():
        ensure(names_out.parent); names_out.write_text("\n".join(names))
    anns_per_img = {}
    for ann in data.get("annotations", []):
        if skip_crowd and ann.get("iscrowd",0)==1: continue
        anns_per_img.setdefault(ann["image_id"], []).append(ann)

    num_boxes = 0
    for img_id, meta in img_by_id.items():
        fn = meta["file_name"]
        w, h = meta.get("width"), meta.get("height")
        if not w or not h:
            try:
                with Image.open(images_dir/fn) as im: w,h = im.size
            except Exception:
                continue
        out_txt = labels_dir/Path(fn).with_suffix(".txt").name
        if out_txt.exists():  # don't rewrite
            continue
        lines=[]
        for ann in anns_per_img.get(img_id, []):
            cat = catid_to_index.get(ann["category_id"])
            if cat is None: continue
            x,y,bw,bh = ann["bbox"]
            cx = (x + bw/2)/w; cy = (y + bh/2)/h
            nw = bw/w; nh = bh/h
            def clip(v): return max(0.0, min(1.0, float(v)))
            cx,cy,nw,nh = map(clip,(cx,cy,nw,nh))
            lines.append(f"{cat} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            num_boxes += 1
        out_txt.write_text("\n".join(lines))
    print(f"[ok] {json_path.name} → {labels_dir} ({num_boxes} boxes)")
    return names

def write_yaml_simple(path_file: Path, ds_root: Path, names: list, img_sub="images", lbl_sub="labels"):
    write_yaml_abs(path_file, ds_root, ds_root/img_sub/"train", ds_root/img_sub/"val", names)

def should_skip_yaml_and_labels(yaml_path: Path, labels_glob: Path):
    return yaml_path.exists() and any(labels_glob.glob("*.txt"))

# ---------------- COCO 2017 ----------------
coco_dir = ROOT/"coco"
coco_img_train = coco_dir/"images/train2017"
coco_img_val   = coco_dir/"images/val2017"
coco_ann_train = coco_dir/"annotations/instances_train2017.json"
coco_ann_val   = coco_dir/"annotations/instances_val2017.json"
coco_yaml = coco_dir/"yolo_coco.yaml"
if coco_img_train.exists() and coco_ann_train.exists():
    if should_skip_yaml_and_labels(coco_yaml, coco_dir/"labels/train2017"):
        print("[ok] COCO YOLO already prepared, skipping")
    else:
        coco_lbl_root = coco_dir/"labels"
        names = coco_to_yolo(coco_ann_train, coco_img_train, coco_lbl_root/"train2017", names_out=coco_dir/"names_coco.txt")
        if coco_ann_val.exists() and coco_img_val.exists():
            _ = coco_to_yolo(coco_ann_val, coco_img_val, coco_lbl_root/"val2017")
        if names:
            write_yaml_abs(coco_yaml, coco_dir, coco_img_train, coco_img_val if coco_img_val.exists() else coco_img_train, names)

# ---------------- LVIS v1 ----------------
lvis_dir = ROOT/"lvis"
lvis_train = lvis_dir/"lvis_v1_train.json"
lvis_val   = lvis_dir/"lvis_v1_val.json"
lvis_yaml = lvis_dir/"yolo_lvis.yaml"
lvis_img_train = lvis_dir/"images/train"
lvis_img_val = lvis_dir/"images/val"

if lvis_train.exists():
    if should_skip_yaml_and_labels(lvis_yaml, lvis_dir/"labels/train2017"):
        print("[ok] LVIS YOLO already prepared, skipping")
    else:
        lvis_lbl_root = lvis_dir/"labels"
        names = coco_to_yolo(lvis_train, lvis_img_train, lvis_lbl_root/"train2017", names_out=lvis_dir/"names_lvis.txt")
        if lvis_val.exists():
            _ = coco_to_yolo(lvis_val, lvis_img_val, lvis_lbl_root/"val2017")
        if names:
            write_yaml_abs(lvis_yaml, lvis_dir, lvis_img_train, lvis_img_val if lvis_img_val.exists() else lvis_img_train, names)

# ---------------- VOC 2007+2012 → YOLO ----------------
voc_root = ROOT/"voc"
voc_yaml = voc_root/"yolo_voc.yaml"
if (voc_root/"VOCdevkit").exists():
    if should_skip_yaml_and_labels(voc_yaml, voc_root/"labels/train"):
        print("[ok] VOC YOLO already prepared, skipping")
    else:
        NAMES_VOC = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
        name_to_id = {n:i for i,n in enumerate(NAMES_VOC)}

        def voc_to_yolo(year_dir: Path, split: str, out_images: Path, out_labels: Path):
            jpeg = year_dir/"JPEGImages"; ann = year_dir/"Annotations"
            setfile = year_dir/"ImageSets/Main"/f"{split}.txt"
            if not setfile.exists(): print(f"[skip] {setfile} not found"); return 0,0
            ids = [l.strip() for l in setfile.read_text().splitlines() if l.strip()]
            out_images.mkdir(parents=True, exist_ok=True)
            out_labels.mkdir(parents=True, exist_ok=True)
            imgs=boxes=0
            for img_id in ids:
                # image
                img_path = (jpeg/f"{img_id}.jpg")
                if not img_path.exists():
                    alt = jpeg/f"{img_id}.png"
                    if alt.exists(): img_path = alt
                    else: continue
                dst_img = out_images/img_path.name
                if not dst_img.exists():
                    try: os.symlink(img_path.resolve(), dst_img)
                    except Exception:
                        shutil.copy2(img_path, dst_img)
                # label
                out_txt = out_labels/img_path.with_suffix(".txt").name
                if out_txt.exists():  # don't rewrite
                    imgs+=1; continue
                xml_path = ann/f"{img_id}.xml"
                lines=[]
                if xml_path.exists():
                    tree = ET.parse(xml_path)
                    W=int(tree.findtext("size/width", "0")); H=int(tree.findtext("size/height","0"))
                    for obj in tree.findall("object"):
                        cls = obj.findtext("name","")
                        if cls not in name_to_id or obj.findtext("difficult")=="1": continue
                        bb = obj.find("bndbox")
                        xmin=float(bb.findtext("xmin","0")); ymin=float(bb.findtext("ymin","0"))
                        xmax=float(bb.findtext("xmax","0")); ymax=float(bb.findtext("ymax","0"))
                        if W<=0 or H<=0 or xmax<=xmin or ymax<=ymin: continue
                        cx=((xmin+xmax)/2)/W; cy=((ymin+ymax)/2)/H
                        bw=(xmax-xmin)/W; bh=(ymax-ymin)/H
                        lines.append(f"{name_to_id[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                        boxes+=1
                out_txt.write_text("\n".join(lines))
                imgs+=1
            return imgs, boxes

        img_train = voc_root/"images/train"; img_val = voc_root/"images/val"
        lbl_train = voc_root/"labels/train"; lbl_val = voc_root/"labels/val"
        img_train.mkdir(parents=True, exist_ok=True)
        img_val.mkdir(parents=True, exist_ok=True)
        lbl_train.mkdir(parents=True, exist_ok=True)
        lbl_val.mkdir(parents=True, exist_ok=True)

        splits = [("VOC2007","trainval"),("VOC2007","test"),("VOC2012","trainval")]
        ti=tb=0
        for year,split in splits:
            ydir = voc_root/"VOCdevkit"/year
            if not ydir.exists(): print(f"[skip] {ydir} missing"); continue
            if split in ("train","trainval"):
                i,b = voc_to_yolo(ydir, split, img_train, lbl_train)
            else:
                i,b = voc_to_yolo(ydir, split, img_val, lbl_val)
            ti+=i; tb+=b

        write_yaml_abs(voc_yaml, voc_root, img_train, img_val, NAMES_VOC)
        print(f"[ok] VOC → YOLO | images:{ti}, boxes:{tb}")

# -------------- Small VOC-like datasets → YOLO --------------
def convert_voc_like(src_root: Path, out_root: Path, train_split: str, val_split: str, names: list):
    jpeg = src_root/"JPEGImages"; ann = src_root/"Annotations"; sets = src_root/"ImageSets"/"Main"
    name_to_id = {n:i for i,n in enumerate(names)}

    def read_ids(split):
        f = sets/f"{split}.txt"
        if f.exists(): return [l.strip() for l in f.read_text().splitlines() if l.strip()]
        return [p.stem for p in jpeg.glob("*.jpg")]

    def do_split(ids, split):
        img_out = out_root/"images"/split; lbl_out = out_root/"labels"/split
        img_out.mkdir(parents=True, exist_ok=True); lbl_out.mkdir(parents=True, exist_ok=True)
        for sid in ids:
            # image
            img = None
            for ext in (".jpg",".png",".jpeg"):
                cand = jpeg/f"{sid}{ext}"
                if cand.exists(): img = cand; break
            if img is None: continue
            dst = img_out/img.name
            if not dst.exists():
                try: dst.symlink_to(img.resolve())
                except Exception:
                    shutil.copy2(img, dst)
            # label
            out_txt = lbl_out/f"{sid}.txt"
            if out_txt.exists():  # don't rewrite
                continue
            xml = ann/f"{sid}.xml"
            lines=[]
            if xml.exists():
                root = ET.parse(xml).getroot()
                sz = root.find("size")
                try:
                    W=float(sz.findtext("width","0")); H=float(sz.findtext("height","0"))
                except Exception: W=H=0
                for obj in root.findall("object"):
                    nm = obj.findtext("name","").strip().lower()
                    if nm not in name_to_id: continue
                    bb = obj.find("bndbox")
                    if bb is None or W<=0 or H<=0: continue
                    try:
                        xmin=float(bb.findtext("xmin","0")); ymin=float(bb.findtext("ymin","0"))
                        xmax=float(bb.findtext("xmax","0")); ymax=float(bb.findtext("ymax","0"))
                    except Exception: continue
                    if xmax<=xmin or ymax<=ymin: continue
                    cx=((xmin+xmax)/2)/W; cy=((ymin+ymax)/2)/H
                    bw=(xmax-xmin)/W; bh=(ymax-ymin)/H
                    lines.append(f"{name_to_id[nm]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            out_txt.write_text("\n".join(lines))

    train_ids = read_ids(train_split); val_ids = read_ids(val_split)
    do_split(train_ids, "train"); do_split(val_ids, "val")

# Clipart1k
clip_src = ROOT/"clipart1k"
if clip_src.exists() and (clip_src/"JPEGImages").exists():
    clip_out = ROOT/"clipart1k_yolo"
    clip_yaml = ROOT/"configs/datasets/clipart1k.yaml"
    if clip_yaml.exists() and nonempty(clip_out/"labels/train"):
        print("[ok] clipart1k YOLO already prepared, skipping")
    else:
        names20 = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
        convert_voc_like(clip_src, clip_out, "train", "val", names20)
        write_yaml_simple(clip_yaml, clip_out, names20)
else:
    print("[FATAL] clipart1k not present after download")

# Watercolor2k & Comic2k
for name in ("watercolor2k","comic2k"):
    src = ROOT/name
    if src.exists() and (src/"JPEGImages").exists():
        out = ROOT/f"{name}_yolo"
        yml = ROOT/f"configs/datasets/{name}.yaml"
        if yml.exists() and nonempty(out/"labels/train"):
            print(f"[ok] {name} YOLO already prepared, skipping")
        else:
            names6 = ["bicycle","bird","car","cat","dog","person"]
            train_split="train"; val_split=("test" if (src/"ImageSets/Main/test.txt").exists() else "val")
            convert_voc_like(src, out, train_split, val_split, names6)
            write_yaml_simple(yml, out, names6)
    else:
        print(f"[FATAL] {name} not present after download")

# PennFudan (mask → boxes)
src = ROOT/"pennfudan"/"PennFudanPed"
if src.exists():
    out = ROOT/"pennfudan_yolo"
    yml = ROOT/"configs/datasets/pennfudan.yaml"
    if yml.exists() and nonempty(out/"labels/train"):
        print("[ok] PennFudan YOLO already prepared, skipping")
    else:
        ensure(out/"images/train"); ensure(out/"labels/train")
        img_dir = src/"PNGImages"; mask_dir = src/"PedMasks"
        for img in sorted(img_dir.glob("*.png")):
            msk = mask_dir/f"{img.stem}_mask.png"
            if not msk.exists(): continue
            dst = out/"images/train"/img.name
            if not dst.exists():
                try: dst.symlink_to(img.resolve())
                except Exception:
                    shutil.copy2(img, dst)
            arr = __import__("numpy").array(Image.open(msk))
            H,W = arr.shape[:2]; lines=[]
            out_txt = out/"labels/train"/f"{img.stem}.txt"
            if out_txt.exists():  # don't rewrite
                continue
            for pid in [p for p in __import__("numpy").unique(arr) if p!=0]:
                ys,xs = (arr==pid).nonzero()
                if xs.size==0 or ys.size==0: continue
                xmin,xmax = xs.min(), xs.max(); ymin,ymax = ys.min(), ys.max()
                if xmax<=xmin or ymax<=ymin: continue
                cx=((xmin+xmax)/2)/W; cy=((ymin+ymax)/2)/H
                bw=(xmax-xmin)/W; bh=(ymax-ymin)/H
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            out_txt.write_text("\n".join(lines))
        write_yaml_abs(yml, out, out/"images/train", out/"images/val", ["person"])
else:
    print("[FATAL] PennFudan dataset not found after download")

# KITTI 2D → YOLO
kitti = ROOT/"kitti"
if (kitti/"image_2").exists() and (kitti/"label_2").exists():
    out = ROOT/"kitti_yolo"
    yml = ROOT/"configs/datasets/kitti.yaml"
    if yml.exists() and nonempty(out/"labels/train"):
        print("[ok] KITTI YOLO already prepared, skipping")
    else:
        ensure(out/"images/train"); ensure(out/"labels/train")
        for img in sorted((kitti/"image_2").glob("*.png")):
            dst = out/"images/train"/img.name
            if not dst.exists():
                try:
                    dst.symlink_to(img.resolve())
                except Exception:
                    shutil.copy2(img, dst)
            try:
                W,H = Image.open(img).size
            except Exception:
                W=H=0
            lblp = kitti/"label_2"/f"{img.stem}.txt"
            out_txt = out/"labels/train"/f"{img.stem}.txt"
            if out_txt.exists():  # don't rewrite
                continue
            lines=[]
            if lblp.exists() and W>0 and H>0:
                for raw in lblp.read_text().splitlines():
                    parts = raw.split()
                    if len(parts)<8: continue
                    cls = parts[0].lower()
                    if cls not in {"car","pedestrian","cyclist"}: continue
                    left,top,right,bottom = map(float, parts[4:8])
                    cx=(left+right)/(2*W); cy=(top+bottom)/(2*H)
                    bw=max(0.0,right-left)/W; bh=max(0.0,bottom-top)/H
                    cid={"car":0,"pedestrian":1,"cyclist":2}[cls]
                    lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            out_txt.write_text("\n".join(lines))
        write_yaml_abs(yml, out, out/"images/train", out/"images/val", ["car","pedestrian","cyclist"])
else:
    print("[FATAL] KITTI expected folders not found after download")

# -------------------- Auto-split when missing val --------------------
def autosplit_if_needed(ds_root: Path):
    img_tr = ds_root/"images/train"; lbl_tr = ds_root/"labels/train"
    img_va = ds_root/"images/val";   lbl_va = ds_root/"labels/val"
    if not img_tr.exists(): return
    ensure(img_va); ensure(lbl_va)
    has_val_imgs = any(img_va.glob("*"))
    if has_val_imgs: return
    imgs = sorted(list(img_tr.glob("*.jpg")) + list(img_tr.glob("*.png")) + list(img_tr.glob("*.jpeg")))
    if not imgs: return
    k = max(1, int(round(len(imgs)*VAL_FRACTION)))
    random.seed(0)
    sample = set(random.sample(imgs, k))
    for p in imgs:
        stem = p.stem
        if p in sample:
            (img_va/p.name).write_bytes(p.read_bytes()); p.unlink(missing_ok=True)
            src_lbl = lbl_tr/f"{stem}.txt"
            dst_lbl = lbl_va/f"{stem}.txt"
            if src_lbl.exists():
                dst_lbl.write_text(src_lbl.read_text()); src_lbl.unlink(missing_ok=True)
            else:
                dst_lbl.write_text("")
    print(f"[ok] autosplit {ds_root.name}: moved {k}/{len(imgs)} images to val")

for ds in ["clipart1k_yolo","watercolor2k_yolo","comic2k_yolo","pennfudan_yolo","kitti_yolo","voc"]:
    root = ROOT/ds
    if root.exists():
        autosplit_if_needed(root)

# ---------------- Final summary ----------------
def count_split(ds_root: Path):
    def cnt_recursive(p: Path):
        return sum(1 for q in p.rglob("*") if q.is_file()) if p.exists() else 0
    img_tr = ds_root/"images"/"train"
    img_va = ds_root/"images"/"val"
    lbl_tr = ds_root/"labels"/"train"
    lbl_va = ds_root/"labels"/"val"
    it = cnt_recursive(img_tr); il = cnt_recursive(lbl_tr)
    vt = cnt_recursive(img_va); vl = cnt_recursive(lbl_va)
    return it, il, vt, vl

print("\n[summary] YOLO datasets (images/labels train|val):")
cands = [ROOT/"coco", ROOT/"lvis", ROOT/"voc", ROOT/"clipart1k_yolo", ROOT/"watercolor2k_yolo", ROOT/"comic2k_yolo", ROOT/"pennfudan_yolo", ROOT/"kitti_yolo"]
ok_any=False
for ds in cands:
    if not ds.exists(): continue
    it,il,vt,vl = count_split(ds)
    print(f"  - {ds.name:<18} train({it:5d}/{il:5d}) | val({vt:5d}/{vl:5d})")
    ok_any=True

if not ok_any:
    print("[FATAL] No YOLO datasets produced"); sys.exit(1)

print("[✓] All conversions complete.")
PY

log "[✓] Dataset preparation finished at $DATA_DIR"