# 复现实验图

新版图8(a)的异质雷达散射截面工况从仓库根目录运行：

    python 实验/run_latest.py --figure8-minmax --fig8a-k3-rcs-m2 2.5

图8(b)(c)的等 RCS 扫描从仓库根目录运行：

    python 实验/run_latest.py --figure8-minmax

五种方法共同采用 8 ms 单目标最低驻留时间和原 16 个噪声种子。加上 `--min-dwell-ms 2` 可用原最低驻留时间运行对比。

图7和图9的原配置通过以下命令复现：

在仓库根目录安装 [Python 依赖](latest/fig8a/requirements.txt)，然后运行：

    python 实验/run_latest.py

两个入口均读取[已记录的 SUMO 轨迹及扫描参数](latest/recursive_feedback/inputs/)。可用 `--output` 指定新的输出目录，用 `--workers` 设置进程数。算法和参数说明见[反馈实验说明](latest/recursive_feedback/README.md)。
