#!/bin/bash
set -e
mkdir -p models

echo "Downloading Facebook FastText LID model (lid.176.bin)..."
if [ ! -f models/lid.176.bin ]; then
    curl -L -o models/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
    echo "Download complete."
else
    echo "Model already exists at models/lid.176.bin"
fi
