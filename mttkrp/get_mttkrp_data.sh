#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"

URLS=(
    "https://frostt-tensors.s3.us-east-2.amazonaws.com/1998DARPA/1998darpa.tns.gz"
    "https://s3.us-east-2.amazonaws.com/frostt/frostt_data/nell/nell-1.tns.gz"
    "https://frostt-tensors.s3.us-east-2.amazonaws.com/FB-M/fb-m.tns.gz"
    "https://s3.us-east-2.amazonaws.com/frostt/frostt_data/nell/nell-2.tns.gz"
)

is_tar() {
    local magic
    magic="$(gzip -dc "$1" 2>/dev/null | dd bs=1 skip=257 count=5 2>/dev/null)"
    [ "$magic" = "ustar" ]
}

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

for url in "${URLS[@]}"; do
    gz_file="$(basename "$url")"

    echo "Downloading $gz_file..."
    wget -q --show-progress -O "$gz_file" "$url"

    echo "Extracting $gz_file..."
    if is_tar "$gz_file"; then
        COPYFILE_DISABLE=1 tar -xzf "$gz_file"
    else
        gunzip -f "$gz_file"
    fi

    rm -f "$gz_file"
done

echo
echo "Verifying expected tensor files..."
for tns in nell-2.tns 1998DARPA.tns fb-m.tns nell-1.tns; do
    if [ ! -s "$DATA_DIR/$tns" ]; then
        echo "ERROR: expected $DATA_DIR/$tns to exist and be non-empty" >&2
        exit 1
    fi
    echo "  OK: $tns ($(du -h "$DATA_DIR/$tns" | cut -f1))"
done

echo "Done. Data available in $DATA_DIR"
