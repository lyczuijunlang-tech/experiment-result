# 复现图7–10(a)

在仓库根目录安装依赖并运行。代码回放已记录的 SUMO 轨迹，图像随运行输出到指定目录的图片子目录。

    python -m pip install -r 实验/requirements.txt
    python 实验/run_latest.py --output constrained_rerun --workers 8

单周期公式和约束验证可以单独执行。

    python -m unittest discover -s 实验/latest/recursive_feedback -p test_ttc_precision.py -v

使用仓库已经发布的 CSV 重新导出图片。

    python 实验/latest/recursive_feedback/plot_figures_7_9.py --published

[正式门限、评价口径与新旧对比](../实验报告.md) · [原始回放输入](latest/recursive_feedback/inputs/)
