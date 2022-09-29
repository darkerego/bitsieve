#!/bin/bash

#case "$@" in

echo 'Compiling ... '
pyinstaller --noconfirm --onefile --hidden-import talib.stream engine.py
echo 'Building ...'
docker build -t bitsieve .