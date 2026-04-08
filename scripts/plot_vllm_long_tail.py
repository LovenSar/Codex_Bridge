#!/usr/bin/env python3
"""
示意：vLLM「Avg generation throughput」类指标的右偏长尾（与此前统计同量级）。
非原始采样数据，仅用于直观展示分布形态。
"""

from __future__ import annotations

import os
import numpy as np
from pathlib import Path

# 无显示器时避免阻塞在 plt.show()
os.environ.setdefault("MPLBACKEND", "Agg")


def main() -> None:
    rng = np.random.default_rng(42)

    # 混合分布：多数为「常态」窗口，少数为「拥堵/排队」极低吞吐，少量为「轻载」高吞吐
    n_main = 7000
    n_slow = 1800
    n_fast = 1200

    # 常态：对数正态，中位数约 12–13 tok/s
    main = rng.lognormal(mean=np.log(12.4), sigma=0.82, size=n_main)

    # 慢尾：极低吞吐（排队、prefill 占比高、多请求等）
    slow = rng.lognormal(mean=np.log(1.2), sigma=0.55, size=n_slow)

    # 偶发高吞吐窗口
    fast = rng.lognormal(mean=np.log(52), sigma=0.28, size=n_fast)

    x = np.concatenate([main, slow, fast])
    x = np.clip(x, 0.1, 130.0)

    # --- 统计量（与讨论中数值同量级即可）
    p10, p50, p90 = np.percentile(x, [10, 50, 90])
    mean = float(np.mean(x))

    try:
        import matplotlib

        matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("需要 matplotlib：pip install matplotlib") from e

    # 尽量用常见中文字体（Windows）
    for name in ("Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC"):
        try:
            plt.rcParams["font.sans-serif"] = [name]
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # 左：直方图 + 核密度（线性坐标）
    ax = axes[0]
    bins = np.linspace(0, min(120, x.max() + 5), 55)
    ax.hist(x, bins=bins, density=True, alpha=0.55, color="#4C72B0", edgecolor="white", linewidth=0.3)
    try:
        from scipy import stats as st

        kde = st.gaussian_kde(x)
        xs = np.linspace(0.1, 120, 500)
        ax.plot(xs, kde(xs), color="#C44E52", linewidth=2, label="KDE 密度")
    except Exception:
        pass

    ax.axvline(p10, color="#55A868", linestyle="--", linewidth=1.5, label=f"P10 = {p10:.1f}")
    ax.axvline(p50, color="#8172B2", linestyle="--", linewidth=1.5, label=f"中位数 = {p50:.1f}")
    ax.axvline(mean, color="#CCB974", linestyle="-", linewidth=2, label=f"均值 = {mean:.1f}")
    ax.axvline(p90, color="#64B5CD", linestyle="--", linewidth=1.5, label=f"P90 = {p90:.1f}")

    ax.axvspan(0, p10, color="red", alpha=0.12, label="左侧慢速尾（体验风险区）")
    ax.set_xlabel("生成吞吐 Avg generation throughput (tokens/s)")
    ax.set_ylabel("概率密度")
    ax.set_title("右偏分布：多数窗口偏慢，少数高吞吐拉高均值")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, 120)

    # 右：等效 ms/token = 1000/x，展示「延迟侧」长尾更明显
    ms = 1000.0 / x
    ms = np.clip(ms, 8, 5000)
    ax2 = axes[1]
    bins2 = np.linspace(8, 500, 60)
    ax2.hist(ms, bins=bins2, density=True, alpha=0.55, color="#DD8452", edgecolor="white", linewidth=0.3)
    try:
        from scipy import stats as st

        kde2 = st.gaussian_kde(ms)
        xs2 = np.linspace(8, 500, 500)
        ax2.plot(xs2, kde2(xs2), color="#4C72B0", linewidth=2, label="KDE 密度")
    except Exception:
        pass

    m10, m50, m90 = np.percentile(ms, [10, 50, 90])  # ms 越大越慢
    ax2.axvline(m10, color="#64B5CD", linestyle="--", linewidth=1.5, label=f"P10（快）≈ {m10:.0f} ms/字")
    ax2.axvline(m50, color="#8172B2", linestyle="--", linewidth=1.5, label=f"中位数 ≈ {m50:.0f} ms/字")
    ax2.axvline(m90, color="#55A868", linestyle="--", linewidth=1.5, label=f"P90（慢）≈ {m90:.0f} ms/字")

    ax2.set_xlabel("等效间隔 1000 / tok/s (ms/token，示意)")
    ax2.set_ylabel("概率密度")
    ax2.set_title("换算到「每字间隔」：慢速长尾在右侧被拉长")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_xlim(8, 500)

    fig.suptitle(
        "vLLM 周期性吞吐的长尾示意（合成数据，与此前均值/分位数同量级）",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()

    out = Path(__file__).resolve().parent.parent / "long_tail_distribution.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"已保存: {out}")
    print(
        f"合成样本统计: mean_tps={mean:.2f}, median={p50:.2f}, "
        f"P10={p10:.2f}, P90={p90:.2f}"
    )

    if os.environ.get("SHOW_PLOT") == "1":
        try:
            plt.show()
        except Exception:
            pass


if __name__ == "__main__":
    main()
