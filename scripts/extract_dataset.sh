#!/bin/bash
# Extract all NIH ChestX-ray14 zip files into images/ directory
# Usage: bash scripts/extract_dataset.sh
# Skips already-extracted zips, safe to re-run

set -e

DATA_DIR="/root/autodl-tmp/data/nih-chest-xrays-data"
IMG_DIR="${DATA_DIR}/images"

mkdir -p "${IMG_DIR}"

echo "============================================"
echo "  Extracting NIH ChestX-ray14 images"
echo "  Source: ${DATA_DIR}"
echo "  Target: ${IMG_DIR}"
echo "============================================"

for ZIP in "${DATA_DIR}"/images_*.zip; do
    if [ ! -f "$ZIP" ]; then
        echo "SKIP (not found): $ZIP"
        continue
    fi
    BASENAME=$(basename "$ZIP")
    echo "Extracting: ${BASENAME} ..."
    python3 -c "
import zipfile, os, sys

zpath = '${ZIP}'
dest = '${IMG_DIR}'

try:
    zf = zipfile.ZipFile(zpath, 'r')
except zipfile.BadZipFile:
    print(f'  WARNING: {zpath} has no central directory, trying streaming extraction...')
    import struct, io
    from PIL import Image
    
    with open(zpath, 'rb') as f:
        count = 0
        fsize = os.path.getsize(zpath)
        while f.tell() < fsize - 30:
            sig = f.read(4)
            if sig != b'PK\x03\x04':
                break
            header = f.read(26)
            ver, flags, method, mtime, mdate, crc32, comp_sz, uncomp_sz, name_len, extra_len = struct.unpack('<HHHHHLLLHH', header)
            fname = f.read(name_len).decode('utf-8', errors='replace')
            f.read(extra_len)
            
            if fname.endswith('.png') and comp_sz > 0:
                basename = os.path.basename(fname)
                outpath = os.path.join(dest, basename)
                if not os.path.exists(outpath):
                    data = f.read(comp_sz)
                    with open(outpath, 'wb') as out:
                        out.write(data)
                    count += 1
                else:
                    f.seek(comp_sz, 1)
            elif comp_sz > 0:
                f.seek(comp_sz, 1)
        print(f'  Extracted {count} images via streaming')
    sys.exit(0)

members = [m for m in zf.namelist() if m.endswith('.png')]
extracted = 0
for m in members:
    basename = os.path.basename(m)
    outpath = os.path.join(dest, basename)
    if os.path.exists(outpath):
        continue
    data = zf.read(m)
    with open(outpath, 'wb') as out:
        out.write(data)
    extracted += 1

zf.close()
print(f'  Extracted {extracted} new images ({len(members)} total in zip)')
"
done

TOTAL=$(ls -1 "${IMG_DIR}"/*.png 2>/dev/null | wc -l)
echo ""
echo "============================================"
echo "  Extraction complete: ${TOTAL} PNG images"
echo "============================================"
