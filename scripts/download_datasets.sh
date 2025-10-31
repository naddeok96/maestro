#!/usr/bin/env bash
set -euo pipefail

# ====================== Config toggles ======================
DATA_DIR_INPUT="${1:-$PWD/data}"
MAKE_YOLO="${MAKE_YOLO:-true}"   # set "false" to skip YOLO conversion
USE_VOC="${USE_VOC:-false}"      # set "true" to download PASCAL VOC 07+12
VAL_FRACTION="${VAL_FRACTION:-0.2}"  # autosplit ratio when only train exists
# ============================================================

# ------------------ Resolve target work dir -----------------
DATA_DIR="$(cd "$(dirname "$DATA_DIR_INPUT")" && mkdir -p "$(basename "$DATA_DIR_INPUT")" && cd "$(basename "$DATA_DIR_INPUT")" && pwd)"
cd "$DATA_DIR"
ROOT_DIR="$(pwd)"

# ------------------ dry-run tiny images ---------------------
if [[ "${DRY_RUN:-false}" == "true" ]]; then
  echo "[dry-run] Creating minimal test images under data/target/images/{train,val}"
  mkdir -p data/target/images/{train,val}
  python - <<'PY'
from pathlib import Path
from PIL import Image
root = Path("data/target/images")
for split in ("train","val"):
    (root/split).mkdir(parents=True, exist_ok=True)
    for i in range(1, 11):
        Image.new("L",(640,640),128).save(root/split/f"img_{i}.jpg","JPEG")
print("[dry-run] (No annotations; YOLO conversion will be skipped.)")
PY
  exit 0
fi

# ======================== COCO 2017 =========================
echo "[*] Preparing COCO 2017"
mkdir -p coco && cd coco
if [[ ! -d images/train2017 || ! -d images/val2017 || ! -f annotations/instances_train2017.json ]]; then
  mkdir -p images
  wget -c http://images.cocodataset.org/zips/train2017.zip
  wget -c http://images.cocodataset.org/zips/val2017.zip
  wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  unzip -n train2017.zip
  unzip -n val2017.zip
  unzip -n annotations_trainval2017.zip
  mkdir -p images
  mv -n train2017 images/train2017 || true
  mv -n val2017 images/val2017   || true
fi
cd "$ROOT_DIR"

# ========================== LVIS ============================
echo "[*] Preparing LVIS v1 (annotations only)"
mkdir -p lvis && cd lvis
[[ ! -f lvis_v1_train.json ]] && wget -c https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip
[[ ! -f lvis_v1_val.json   ]] && wget -c https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip
[[ -f lvis_v1_train.json.zip ]] && unzip -n lvis_v1_train.json.zip
[[ -f lvis_v1_val.json.zip   ]] && unzip -n lvis_v1_val.json.zip
cd "$ROOT_DIR"

# ==================== Target dataset slot ===================
echo "[*] Target dataset placeholder"
mkdir -p target
cat <<EOF
Place your COCO-format target dataset under: $DATA_DIR/target
  Expected structure:
    target/images/train/*.jpg, target/annotations/instances_train.json
    target/images/val/*.jpg,   target/annotations/instances_val.json
EOF

# ======================== PASCAL VOC ========================
echo "[*] Preparing PASCAL VOC 2007+2012 (optional)"
if [[ "$USE_VOC" == "true" ]]; then
  mkdir -p voc && cd voc

  download_voc() {
    local outfile="$1"; shift
    local urls=("$@")
    if [[ -f "$outfile" ]]; then
      if [[ $(stat -c%s "$outfile" 2>/dev/null || echo 0) -gt 10000000 ]] && tar -tf "$outfile" >/dev/null 2>&1; then
        echo "[ok] Using existing $outfile"
        return 0
      else
        echo "[!] $outfile exists but invalid; re-downloading"
        rm -f "$outfile"
      fi
    fi
    for u in "${urls[@]}"; do
      echo "[*] Trying $u"
      if [[ "$u" =~ ^https?://pjreddie\.com/ ]]; then
        if curl -fL --retry 3 --retry-delay 2 \
           -A "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36" \
           -o "$outfile" "$u"; then :; else echo "[!] Failed: $u"; continue; fi
      else
        if ! wget -c -O "$outfile" "$u"; then echo "[!] Failed: $u"; continue; fi
      fi
      local sz; sz=$(stat -c%s "$outfile" 2>/dev/null || echo 0)
      if [[ "$sz" -le 10000000 ]]; then echo "[!] Too small; discarding"; rm -f "$outfile"; continue; fi
      if ! tar -tf "$outfile" >/dev/null 2>&1; then echo "[!] Tar invalid; discarding"; rm -f "$outfile"; continue; fi
      echo "[ok] Fetched valid $outfile"; return 0
    done
    return 1
  }

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

  for t in VOCtrainval_06-Nov-2007.tar VOCtest_06-Nov-2007.tar VOCtrainval_11-May-2012.tar; do
    echo "[*] Extracting $t"; tar -xf "$t"
  done
  cd "$ROOT_DIR"
else
  echo "[i] USE_VOC=false — skipping VOC download"
fi

# =================== YOLO conversions core ==================
if [[ "$MAKE_YOLO" == "true" ]]; then
  echo "[*] Converting available datasets to YOLO"

  python - <<'PY'
import json, yaml, os, shutil, random
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(".").resolve()
VAL_FRACTION = float(os.environ.get("VAL_FRACTION","0.2"))

def ensure(p: Path): p.mkdir(parents=True, exist_ok=True)

def write_yaml_abs(yaml_path: Path, ds_root: Path, train_images: Path, val_images: Path, names):
    ds_root = ds_root.resolve()
    y = {
        "path": str(ds_root),
        "train": str(train_images.resolve()),
        "val": str(val_images.resolve()),
        "names": names or [],
    }
    ensure(yaml_path.parent)
    yaml_path.write_text(yaml.dump(y, sort_keys=False))
    print(f"[ok] wrote {yaml_path}")

# ---------------- COCO/LVIS helper ----------------
def coco_to_yolo(json_path: Path, images_dir: Path, labels_dir: Path, names_out: Path=None, skip_crowd=True):
    if not json_path.exists():
        print(f"[skip] {json_path} missing"); return None
    ensure(labels_dir)
    data = json.loads(json_path.read_text())
    # images index (robust to missing file_name)
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
    if names_out is not None:
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
                from PIL import Image
                with Image.open(images_dir/fn) as im: w,h = im.size
            except Exception: 
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
        (labels_dir/Path(fn).with_suffix(".txt").name).write_text("\n".join(lines))
    print(f"[ok] {json_path.name} → {labels_dir} ({num_boxes} boxes)")
    return names

# ---------------- COCO 2017 ----------------
coco_dir = ROOT/"coco"
coco_img_train = coco_dir/"images/train2017"
coco_img_val   = coco_dir/"images/val2017"
coco_ann_train = coco_dir/"annotations/instances_train2017.json"
coco_ann_val   = coco_dir/"annotations/instances_val2017.json"
if coco_img_train.exists() and coco_ann_train.exists():
    coco_lbl_root = coco_dir/"labels"
    names = coco_to_yolo(coco_ann_train, coco_img_train, coco_lbl_root/"train2017", names_out=coco_dir/"names_coco.txt")
    if coco_ann_val.exists() and coco_img_val.exists():
        _ = coco_to_yolo(coco_ann_val, coco_img_val, coco_lbl_root/"val2017")
    if names:
        write_yaml_abs(coco_dir/"yolo_coco.yaml", coco_dir, coco_img_train, coco_img_val if coco_img_val.exists() else coco_img_train, names)

# ---------------- LVIS v1 ----------------
lvis_dir = ROOT/"lvis"
lvis_train = lvis_dir/"lvis_v1_train.json"
lvis_val   = lvis_dir/"lvis_v1_val.json"
if lvis_train.exists() and coco_img_train.exists():
    lvis_lbl_root = lvis_dir/"labels"
    names = coco_to_yolo(lvis_train, coco_img_train, lvis_lbl_root/"train2017", names_out=lvis_dir/"names_lvis.txt")
    if lvis_val.exists() and coco_img_val.exists():
        _ = coco_to_yolo(lvis_val, coco_img_val, lvis_lbl_root/"val2017")
    if names:
        write_yaml_abs(lvis_dir/"yolo_lvis.yaml", lvis_dir, coco_img_train, coco_img_val if coco_img_val.exists() else coco_img_train, names)

# ---------------- Target (COCO-style) ----------------
tgt = ROOT/"target"
tgt_img_train = tgt/"images/train"
tgt_img_val   = tgt/"images/val"
tgt_ann_train = tgt/"annotations/instances_train.json"
tgt_ann_val   = tgt/"annotations/instances_val.json"
if tgt.exists() and tgt_ann_train.exists() and tgt_img_train.exists():
    tgt_lbl_root = tgt/"labels"
    names = coco_to_yolo(tgt_ann_train, tgt_img_train, tgt_lbl_root/"train", names_out=tgt/"names_target.txt")
    if tgt_ann_val.exists() and tgt_img_val.exists():
        _ = coco_to_yolo(tgt_ann_val, tgt_img_val, tgt_lbl_root/"val")
    # derive names if missing
    if not names and (tgt/"names_target.txt").exists():
        names = (tgt/"names_target.txt").read_text().splitlines()
    write_yaml_abs(tgt/"yolo_target.yaml", tgt, tgt_img_train, tgt_img_val if tgt_img_val.exists() else tgt_img_train, names or [])

# ---------------- VOC → YOLO ----------------
voc_root = ROOT/"voc"
if (voc_root/"VOCdevkit").exists():
    NAMES = [
        "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair",
        "cow","diningtable","dog","horse","motorbike","person","pottedplant",
        "sheep","sofa","train","tvmonitor"
    ]
    name_to_id = {n:i for i,n in enumerate(NAMES)}

    def voc_to_yolo(year_dir: Path, split: str, out_images: Path, out_labels: Path):
        jpeg = year_dir/"JPEGImages"; ann = year_dir/"Annotations"
        setfile = year_dir/"ImageSets/Main"/f"{split}.txt"
        if not setfile.exists(): print(f"[skip] {setfile} not found"); return 0,0
        ids = [l.strip() for l in setfile.read_text().splitlines() if l.strip()]
        ensure(out_images); ensure(out_labels)
        imgs=boxes=0
        for img_id in ids:
            img_path = (jpeg/f"{img_id}.jpg")
            if not img_path.exists():
                alt = jpeg/f"{img_id}.png"
                if alt.exists(): img_path = alt
                else: continue
            dst_img = out_images/img_path.name
            try: os.symlink(img_path.resolve(), dst_img)
            except Exception:
                if not dst_img.exists(): shutil.copy2(img_path, dst_img)

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
            (out_labels/img_path.with_suffix(".txt").name).write_text("\n".join(lines))
            imgs+=1
        return imgs, boxes

    img_train = voc_root/"images/train"; img_val = voc_root/"images/val"
    lbl_train = voc_root/"labels/train"; lbl_val = voc_root/"labels/val"
    ensure(img_train); ensure(img_val); ensure(lbl_train); ensure(lbl_val)

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

    write_yaml_abs(voc_root/"yolo_voc.yaml", voc_root, img_train, img_val, NAMES)
    print(f"[ok] VOC → YOLO | images:{ti}, boxes:{tb}")

# ---------------- Small datasets helpers ----------------
def convert_voc_like(src_root: Path, out_root: Path, train_split: str, val_split: str, names: list):
    jpeg = src_root/"JPEGImages"; ann = src_root/"Annotations"; sets = src_root/"ImageSets"/"Main"
    name_to_id = {n:i for i,n in enumerate(names)}
    def read_ids(split):
        f = sets/f"{split}.txt"
        if f.exists(): return [l.strip() for l in f.read_text().splitlines() if l.strip()]
        return [p.stem for p in jpeg.glob("*.jpg")]
    def do_split(ids, split):
        img_out = out_root/"images"/split; lbl_out = out_root/"labels"/split
        ensure(img_out); ensure(lbl_out)
        for sid in ids:
            img = jpeg/f"{sid}.jpg"
            if not img.exists():
                alt = jpeg/f"{sid}.png"
                if alt.exists(): img = alt
                else: continue
            dst = img_out/img.name
            try: dst.symlink_to(img.resolve())
            except Exception:
                if not dst.exists(): shutil.copy2(img, dst)
            xml = ann/f"{sid}.xml"
            lines=[]
            if xml.exists():
                root = ET.parse(xml).getroot()
                sz = root.find("size")
                if sz is not None:
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
            (lbl_out/f"{sid}.txt").write_text("\n".join(lines))
    train_ids = read_ids(train_split); val_ids = read_ids(val_split)
    do_split(train_ids, "train"); do_split(val_ids, "val")

def write_yaml_simple(path_file: Path, ds_root: Path, names: list):
    write_yaml_abs(path_file, ds_root, ds_root/"images/train", ds_root/"images/val", names)

# Clipart1k (VOC 20 classes)
clip_src = ROOT/"clipart1k"
if clip_src.exists() and (clip_src/"JPEGImages").exists():
    clip_out = ROOT/"clipart1k_yolo"
    names20 = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
               "diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
    convert_voc_like(clip_src, clip_out, "train", "val")
    write_yaml_simple(ROOT/"configs/datasets/clipart1k.yaml", clip_out, names20)
else:
    print("[i] clipart1k not present (manual download needed)")

# Watercolor2k (6 classes), Comic2k (6 classes) — map test→val
for name in ("watercolor2k","comic2k"):
    src = ROOT/name
    if src.exists() and (src/"JPEGImages").exists():
        out = ROOT/f"{name}_yolo"
        names6 = ["bicycle","bird","car","cat","dog","person"]
        train_split="train"; val_split=("test" if (src/"ImageSets/Main/test.txt").exists() else "val")
        convert_voc_like(src, out, train_split, val_split, names6)
        write_yaml_simple(ROOT/f"configs/datasets/{name}.yaml", out, names6)
    else:
        print(f"[i] {name} not present (manual download needed)")

# PennFudan (mask → boxes), will create train only; autosplit later
src = ROOT/"pennfudan"/"PennFudanPed"
if src.exists():
    import numpy as np
    out = ROOT/"pennfudan_yolo"
    ensure(out/"images/train"); ensure(out/"labels/train")
    img_dir = src/"PNGImages"; mask_dir = src/"PedMasks"
    for img in sorted(img_dir.glob("*.png")):
        msk = mask_dir/f"{img.stem}_mask.png"
        if not msk.exists(): continue
        dst = out/"images/train"/img.name
        try: dst.symlink_to(img.resolve())
        except Exception:
            if not dst.exists(): shutil.copy2(img, dst)
        arr = np.array(__import__("PIL").Image.open(msk))
        H,W = arr.shape[:2]; lines=[]
        for pid in [p for p in np.unique(arr) if p!=0]:
            ys,xs = (arr==pid).nonzero()
            if xs.size==0 or ys.size==0: continue
            xmin,xmax = xs.min(), xs.max(); ymin,ymax = ys.min(), ys.max()
            if xmax<=xmin or ymax<=ymin: continue
            cx=((xmin+xmax)/2)/W; cy=((ymin+ymax)/2)/H
            bw=(xmax-xmin)/W; bh=(ymax-ymin)/H
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (out/"labels/train"/f"{img.stem}.txt").write_text("\n".join(lines))
    write_yaml_abs(ROOT/"configs/datasets/pennfudan.yaml", out, out/"images/train", out/"images/val", ["person"])
else:
    print("[i] pennfudan not present (run downloader above)")

# KITTI 2D — expect image_2 + label_2 (txt), create train then autosplit
kitti = ROOT/"kitti"
if (kitti/"image_2").exists() and (kitti/"label_2").exists():
    from PIL import Image
    out = ROOT/"kitti_yolo"
    ensure(out/"images/train"); ensure(out/"labels/train")
    for img in sorted((kitti/"image_2").glob("*.png")):
        dst = out/"images/train"/img.name
        try: dst.symlink_to(img.resolve())
        except Exception:
            if not dst.exists(): shutil.copy2(img, dst)
        try:
            W,H = Image.open(img).size
        except Exception:
            W=H=0
        lblp = kitti/"label_2"/f"{img.stem}.txt"
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
        (out/"labels/train"/f"{img.stem}.txt").write_text("\n".join(lines))
    write_yaml_abs(ROOT/"configs/datasets/kitti.yaml", out, out/"images/train", out/"images/val", ["car","pedestrian","cyclist"])
else:
    print("[i] kitti image_2/label_2 not present (manual download needed)")

# -------------------- Auto-split step --------------------
# For any YOLO dataset that lacks a real val split, move VAL_FRACTION of train to val (with labels).
def autosplit_if_needed(ds_root: Path):
    img_tr = ds_root/"images/train"; lbl_tr = ds_root/"labels/train"
    img_va = ds_root/"images/val";   lbl_va = ds_root/"labels/val"
    if not img_tr.exists(): return
    ensure(img_va); ensure(lbl_va)
    has_val_imgs = any(img_va.glob("*"))
    if has_val_imgs: return
    imgs = sorted(list(img_tr.glob("*.jpg")) + list(img_tr.glob("*.png")))
    if not imgs: return
    k = max(1, int(round(len(imgs)*VAL_FRACTION)))
    random.seed(0)
    sample = set(random.sample(imgs, k))
    for p in imgs:
        stem = p.stem
        (img_va/p.name).write_bytes(p.read_bytes()); p.unlink()
        src_lbl = lbl_tr/f"{stem}.txt"
        dst_lbl = lbl_va/f"{stem}.txt"
        if src_lbl.exists():
            dst_lbl.write_text(src_lbl.read_text()); src_lbl.unlink()
        else:
            dst_lbl.write_text("")  # ensure label exists
    print(f"[ok] autosplit {ds_root.name}: moved {k}/{len(imgs)} images to val")

# Attempt autosplit on datasets that we expect may lack val at this point
for ds in ["pennfudan_yolo","kitti_yolo"]:
    root = ROOT/ds
    if root.exists():
        autosplit_if_needed(root)

# Also ensure val exists for target if only train present
target_yolo = ROOT/"target"
if (target_yolo/"labels/train").exists():
    autosplit_if_needed(target_yolo)

# ---------------- Final sanity & summary ----------------
def count_split(ds_root: Path):
    def cnt(p): return sum(1 for _ in p.glob("*")) if p.exists() else 0
    it = cnt(ds_root/"images/train"); il = cnt(ds_root/"labels/train")
    vt = cnt(ds_root/"images/val");   vl = cnt(ds_root/"labels/val")
    return it,il,vt,vl

print("\n[summary] YOLO datasets (images/labels train|val):")
cands = [
    ROOT/"coco",
    ROOT/"lvis",
    ROOT/"target",
    ROOT/"voc",
    ROOT/"clipart1k_yolo",
    ROOT/"watercolor2k_yolo",
    ROOT/"comic2k_yolo",
    ROOT/"pennfudan_yolo",
    ROOT/"kitti_yolo",
]
for ds in cands:
    if not ds.exists(): continue
    # infer yolo subpaths
    if (ds/"images/train").exists():
        base = ds
    elif (ds/"images"/"train2017").exists():  # COCO/LVIS images live under coco/images/*
        base = ds
    else:
        continue
    it,il,vt,vl = count_split(base)
    print(f"  - {ds.name:<18} train({it:5d}/{il:5d}) | val({vt:5d}/{vl:5d})")

print("[✓] YOLO conversion complete")
PY

else
  echo "[i] MAKE_YOLO=false — skipping YOLO conversion"
fi

echo "[✓] Dataset preparation finished at $DATA_DIR"

# ================= Small dataset download helpers =================
download_clipart1k() {
  local OUTDIR="clipart1k"; mkdir -p "$OUTDIR"
  if [[ -n "${CLIPART1K_URL:-}" ]]; then
    echo "[*] Fetching Clipart1k from $CLIPART1K_URL"
    wget -c -O "$OUTDIR/clipart1k.zip" "$CLIPART1K_URL" && unzip -n "$OUTDIR/clipart1k.zip" -d "$OUTDIR" || echo "[!] Failed Clipart1k"
  else
    echo "[i] Clipart1k needs a manual download; place VOC-style dirs into $OUTDIR/{JPEGImages,Annotations,ImageSets}"
  fi
}
download_watercolor2k() {
  local OUTDIR="watercolor2k"; mkdir -p "$OUTDIR"
  echo "[*] Watercolor2k page: https://opendatalab.com/OpenDataLab/Watercolor2k/download"
  echo "[i] After download, place VOC-style dirs into: $OUTDIR/{JPEGImages,Annotations,ImageSets}"
}
download_comic2k() {
  local OUTDIR="comic2k"; mkdir -p "$OUTDIR"
  echo "[*] Comic2k page: https://opendatalab.com/OpenDataLab/Comic2k/download"
  echo "[i] After download, place VOC-style dirs into: $OUTDIR/{JPEGImages,Annotations,ImageSets}"
}
download_pennfudan() {
  local OUTDIR="pennfudan"; mkdir -p "$OUTDIR"
  echo "[*] Downloading Penn-Fudan from UPenn"
  wget -c -O "$OUTDIR/PennFudanPed.zip" "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip" && unzip -n "$OUTDIR/PennFudanPed.zip" -d "$OUTDIR" || echo "[!] PennFudan manual download required"
}
download_kitti() {
  local OUTDIR="kitti"; mkdir -p "$OUTDIR"
  echo "[*] KITTI: https://www.cvlibs.net/datasets/kitti/"
  echo "[i] Download 2D object detection 'data_object_image_2' and 'data_object_label_2' into $OUTDIR/image_2 and $OUTDIR/label_2"
}

# Convenience: fetch small public sets (still requires manual for some)
prep_small_sets() {
  echo "[*] Preparing small public datasets (optional/manual)"
  pushd "$ROOT_DIR" >/dev/null
  download_clipart1k
  download_watercolor2k
  download_comic2k
  download_pennfudan
  download_kitti
  popd >/dev/null
}

prep_small_sets
