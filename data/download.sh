#!/bin/bash

if ! command -v gdown &> /dev/null
then
    echo "未检测到 gdown，开始安装..."
    pip install gdown
fi

echo "开始下载文件夹内容到当前目录..."
gdown --folder "https://drive.google.com/drive/folders/1-yU3rmGhoimyCNx7oasBRpuDKSVqJNBa?usp=drive_link" -O .
gdown --folder "https://drive.google.com/drive/folders/1-sGXdTuTLKU8H7IE1uGmvUBmko-y395z?usp=drive_link" -O .

echo "✅ 文件夹下载完成"