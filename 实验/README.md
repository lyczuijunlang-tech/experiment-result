# 复现图7–9

在仓库根目录安装 [Python 依赖](latest/fig8a/requirements.txt)，然后运行：

    python 实验/run_latest.py

入口读取[已记录的 SUMO 轨迹及扫描参数](latest/recursive_feedback/inputs/)，进行 16 组噪声回放，并导出图7–9。可用参数  --output  指定新的输出目录，用  --workers  设置进程数。算法和参数说明见[反馈实验说明](latest/recursive_feedback/README.md)。
