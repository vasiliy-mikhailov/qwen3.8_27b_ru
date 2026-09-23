#!/bin/sh
# Publish the r1 artifact once the owner has logged in with `hf auth login`.
# Refuses to upload under any account other than the one agreed on.
set -u
HF=$HOME/.local/bin/hf
REPO=vasiliy-mikhailov/Qwen3.8-27B-nvfp4-ru-NInfer
DIR=/home/vmihaylov/hf-r1
until $HF auth whoami >/dev/null 2>&1; do sleep 20; done
WHO=$($HF auth whoami 2>&1)
echo "$WHO" | grep -qw "vasiliy-mikhailov" || { echo "logged in as: $WHO -- not vasiliy-mikhailov, not uploading"; exit 1; }
echo "$(date +%H:%M:%S) logged in, uploading to $REPO"
(cd $DIR && sha256sum -c SHA256SUMS) || { echo "checksum mismatch, not uploading"; exit 1; }
$HF upload $REPO $DIR . --repo-type model \
  --commit-message "Qwen3.8-27B NVFP4 for NInfer, MLP re-rounded with AutoRound on Russian data (r1)" 2>&1 | tail -5
echo "$(date +%H:%M:%S) UPLOAD_EXIT=$?"
