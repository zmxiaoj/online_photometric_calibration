#!/bin/zsh

# 定义results文件夹路径
RESULTS_DIR="/home/zmxj/code/online_photometric_calibration/results"

# 检查results文件夹是否存在
if [ -d "$RESULTS_DIR" ]; then
    # 清空results文件夹中的内容
    rm -rf "$RESULTS_DIR"/*
else
    # 如果results文件夹不存在，则创建它
    mkdir -p "$RESULTS_DIR"
fi

# 创建forward和backward文件夹
mkdir -p "$RESULTS_DIR/forward"
mkdir -p "$RESULTS_DIR/backward"

echo "Results folder has been cleared and forward/backward folders have been created."