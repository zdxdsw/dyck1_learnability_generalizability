"""
Plot first-violation statistics for negative Dyck validation/testing examples.

Each row in the output figure corresponds to one <VARIANT>/fix_d_{d}.json file.
Columns:
  1. even-length negative examples: P_FV histogram
  2. odd-length negative examples:  P_FV histogram
  3. even-length negative examples: R_FV histogram
  4. odd-length negative examples:  R_FV histogram
"""

import argparse, json, re
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter # help format the y-axis as percentages
from matplotlib.transforms import offset_copy
import numpy as np
from tqdm import tqdm

TRAIN_SOURCES = [
    {
        "variant": "ORIGINAL",
        "root": "/data/yingshac/tmlr/train",
    },
    {
        "variant": "SHORTER",
        "root": "/data/yingshac/tmlr/shorter/train",
    },
]

VALIDATION_SOURCES = [
    {
        "variant": "ORIGINAL",
        "root": "/data/yingshac/tmlr/validation",
    },
    {
        "variant": "SHORTER",
        "root": "/data/yingshac/tmlr/shorter/validation",
    },
]

TEST_SOURCES = [
    {
        "variant": "ORIGINAL",
        "root": "/data/yingshac/tmlr/test",
    },
    {
        "variant": "SHORTER",
        "root": "/data/yingshac/tmlr/shorter/test",
    },
]


def get_first_violation_statistics(s: str):
    """Return (P_FV, R_FV) for a bracket string."""
    if len(s) == 0: raise ValueError("Input string must be nonempty.")
    c_open, c_closed = 0, 0
    p_fv = len(s)
    r_fv = None

    for p, x in enumerate(s):
        if x == "(":
            c_open += 1
        elif x == ")":
            c_closed += 1
        else:
            raise ValueError(f"Unexpected character {x!r} in string {s!r}")

        if c_closed > c_open:
            p_fv = p
            r_fv = 2 * (c_open / (p + 1) - 0.5)
            break

    if r_fv is None: # This corresponds to the <eos> position case.
        r_fv = 2 * (c_open / len(s) - 0.5)

    return p_fv, r_fv


def extract_d(path: Path):
    """Extract d from filenames like fix_d_3.json. Unknown d goes last."""
    match = re.search(r"fix_d_(\d+)\.json$", path.name)
    return int(match.group(1)) if match else float("inf")


def iter_files(sources):
    """Yield dictionaries describing all fix_d_{d}.json files."""
    rows = []

    for source in sources:
        variant = source["variant"]
        root = Path(source["root"])
        files = sorted(root.glob("fix_d_*.json"), key=extract_d)

        for path in files:
            rows.append(
                {
                    "variant": variant,
                    "root": root,
                    "path": path,
                    "d": extract_d(path),
                }
            )

    return rows


def load_negative_strings(path: Path):
    """Load all negative strings from one validation/testing json file."""
    with path.open("r") as f:
        data = json.load(f)

    if "neg" not in data:
        raise KeyError(f"{path} does not contain a top-level 'neg' field.")

    strings = []
    for length_key, examples in data["neg"].items():
        if not isinstance(examples, list):
            raise TypeError(
                f"{path}: data['neg'][{length_key!r}] should be a list, "
                f"but got {type(examples).__name__}."
            )
        strings.extend(examples)

    return strings


def compute_stats(strings):
    """Split negative strings by length parity and compute P_FV/R_FV."""
    stats = {
        "even": {"p_fv": [], "r_fv": []},
        "odd": {"p_fv": [], "r_fv": []},
    }

    for s in strings:
        p_fv, r_fv = get_first_violation_statistics(s)
        parity = "even" if len(s) % 2 == 0 else "odd"
        stats[parity]["p_fv"].append(p_fv)
        stats[parity]["r_fv"].append(r_fv)

    return stats

def add_variant_separator(fig, axes, rows, variant_a="ORIGINAL", variant_b="SHORTER"):
    separator_idx = None

    for i in range(len(rows) - 1):
        if rows[i]["variant"] == variant_a and rows[i + 1]["variant"] == variant_b:
            separator_idx = i
            break

    if separator_idx is None:
        return

    upper_row_axes = axes[separator_idx, :]
    lower_row_axes = axes[separator_idx + 1, :]

    upper_bottom = min(ax.get_position().y0 for ax in upper_row_axes)
    lower_top = max(ax.get_position().y1 for ax in lower_row_axes)

    y = (upper_bottom + lower_top) / 2

    x0 = min(ax.get_position().x0 for ax in axes.ravel())
    x1 = max(ax.get_position().x1 for ax in axes.ravel())

    line = plt.Line2D(
        [x0, x1],
        [y, y],
        transform=fig.transFigure,
        color="black",
        linewidth=1.5,
        alpha=0.5,
    )
    fig.add_artist(line)

def add_gaussian_curve(ax, values, bins, *, color="black", linewidth=1.5):
    """
    Overlay a Gaussian curve with mean/std estimated from values.

    Assumes the histogram is normalized so bar heights sum to 1
    across bins, i.e. weights=np.ones(len(values)) / len(values).
    Therefore the Gaussian PDF is multiplied by bin_width so that
    the curve is on the same 'percentage per bin' scale.
    """
    if len(values) < 2:
        return

    values = np.asarray(values, dtype=float)

    mu = values.mean()
    sigma = values.std()

    if sigma == 0:
        ax.axvline(mu, color=color, linewidth=linewidth, linestyle="--", alpha=0.8)
        return

    x_min = bins[0]
    x_max = bins[-1]
    x = np.linspace(x_min, x_max, 300)

    pdf = (
        1.0 / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    )

    # Match normalized histogram scale: probability mass per bin.
    # Since the histogram y-axis is percentage per bin, 
    # I should plot a Gaussian curve scaled by the bin width:
    bin_width = np.mean(np.diff(bins))
    y = pdf * bin_width

    ax.plot(x, y, color=color, linewidth=linewidth, linestyle="--", alpha=0.9)

def integer_bins(values):
    """
        make histogram bin edges so that each 
        integer value of P_FV gets its own centered bin.
    """

    lo = min(values)
    hi = max(values)
    return np.arange(lo - 0.5, hi + 1.5, 1.0)


def plot_one_hist(ax, values, *, bins, color, alpha, title, xlabel):
    values = np.asarray(values)
    weights = np.ones(len(values)) / len(values)
    ax.hist(values, bins=bins, weights=weights, color=color, alpha=alpha)
    add_gaussian_curve(
        ax,
        values,
        bins,
        color="black",
        linewidth=1.5,
    )

    if "R_" in xlabel:
        neg_values = values[values < 0]
        pos_values = values[values > 0]

        # x is in data coordinates; y is in axes coordinates -> put text annotation closer to x=0
        trans = ax.get_xaxis_transform()

        label_margin = 8  # points away from each vertical line
        top_row_y = 0.95
        second_row_y = 0.86

        if len(neg_values) > 0:
            min_neg = np.min(neg_values)
            max_neg = np.max(neg_values)

            ax.axvline(max_neg, color="black", linestyle="--", linewidth=1, alpha=0.5)
            text_trans = offset_copy(trans, fig=ax.figure, x=-label_margin, y=0, units="points")
            ax.text(
                max_neg,
                top_row_y,
                f"max neg = {max_neg:.3f}",
                transform=text_trans,
                ha="right",
                va="top",
                fontsize=8,
            )
            ax.text(
                0.02,
                second_row_y,
                f"min neg = {min_neg:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
            )

        if len(pos_values) > 0:
            min_pos = np.min(pos_values)
            max_pos = np.max(pos_values)

            ax.axvline(min_pos, color="black", linestyle="--", linewidth=1, alpha=0.5)
            text_trans = offset_copy(trans, fig=ax.figure, x=label_margin, y=0, units="points")
            ax.text(
                min_pos,
                top_row_y,
                f"min pos = {min_pos:.3f}",
                transform=text_trans,
                ha="left",
                va="top",
                fontsize=8,
            )
            ax.text(
                0.98,
                second_row_y,
                f"max pos = {max_pos:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Percentage")


def make_plot(sources, output_path, dpi=200):
    rows = iter_files(sources)

    if len(rows) == 0:
        roots = ", ".join(source["root"] for source in sources)
        raise FileNotFoundError(f"No fix_d_*.json files found under: {roots}")

    ### Calculating statistics
    all_stats = []
    for row in tqdm(rows, desc="Calculating statistics"):
        strings = load_negative_strings(row["path"])
        stats = compute_stats(strings)
        all_stats.append(stats)

    nrows = len(rows)

    ### Make Plots
    # Find where ORIGINAL switches to SHORTER.
    separator_after = None
    for i in range(len(rows) - 1):
        if rows[i]["variant"] == "ORIGINAL" and rows[i + 1]["variant"] == "SHORTER":
            separator_after = i
            break

    has_separator = separator_after is not None

    if has_separator:
        # Add one extra GridSpec row used only as visual whitespace/separator.
        grid_nrows = nrows + 1

        height_ratios = []
        for i in range(nrows):
            height_ratios.append(1.0)
            if i == separator_after:
                height_ratios.append(0.03)  # tune this gap size
    else:
        grid_nrows = nrows
        height_ratios = [1.0] * nrows

    fig = plt.figure(figsize=(18, max(2.8 * nrows, 3.2)))

    gs = fig.add_gridspec(
        nrows=grid_nrows,
        ncols=4,
        height_ratios=height_ratios,
    )

    axes = np.empty((nrows, 4), dtype=object)

    plot_row = 0
    for i in range(nrows):
        for j in range(4):
            axes[i, j] = fig.add_subplot(gs[plot_row, j])

        plot_row += 1
        if has_separator and i == separator_after:
            plot_row += 1
    
    ### Plotting histograms
    col_titles = [
        "even length: $P_{FV}$",
        "odd length: $P_{FV}$",
        "even length: $R_{FV}$",
        "odd length: $R_{FV}$",
    ]

    r_bins = np.linspace(-1.0, 1.0, 41)  # 40 bins over [-1, 1]

    for i, (row, stats) in tqdm(
        enumerate(zip(rows, all_stats)), 
        desc="Plotting histograms",
        total=nrows,
    ):
        even_p = stats["even"]["p_fv"]
        odd_p = stats["odd"]["p_fv"]
        even_r = stats["even"]["r_fv"]
        odd_r = stats["odd"]["r_fv"]

        row_label = f"{row['variant']}/fix_d_{row['d']}.json"

        # Use parity-specific integer bins so the P_FV axis remains clean.
        even_p_bins = integer_bins(even_p)
        odd_p_bins = integer_bins(odd_p)

        plot_one_hist(
            axes[i, 0],
            even_p,
            bins=even_p_bins,
            color="green",
            alpha=0.5,
            title=col_titles[0],
            xlabel="$P_{FV}$",
        )
        plot_one_hist(
            axes[i, 1],
            odd_p,
            bins=odd_p_bins,
            color="blue",
            alpha=0.5,
            title=col_titles[1],
            xlabel="$P_{FV}$",
        )
        plot_one_hist(
            axes[i, 2],
            even_r,
            bins=r_bins,
            color="green",
            alpha=0.5,
            title=col_titles[2],
            xlabel="$R_{FV}$",
        )
        plot_one_hist(
            axes[i, 3],
            odd_r,
            bins=r_bins,
            color="blue",
            alpha=0.5,
            title=col_titles[3],
            xlabel="$R_{FV}$",
        )

        axes[i, 0].set_ylabel(f"{row_label}\nPercentage")
        axes[i, 2].set_xlim(-1.0, 1.0)
        axes[i, 3].set_xlim(-1.0, 1.0)  

        for j in range(4):
            axes[i, j].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            axes[i, j].grid(axis="y", alpha=0.25)

    fig.tight_layout()

    # add_variant_separator(...) should come after fig.tight_layout(), because tight_layout() changes the subplot positions.
    # This will draw one thick black horizontal line between the ORIGINAL chunk and the SHORTER chunk.
    add_variant_separator(fig, axes, rows)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


def main():
    split = "train"  # "validation", "test", or "train"
    sources = eval(f"{split.upper()}_SOURCES")
    
    make_plot(
        sources=sources,
        output_path=f"{split}_negative_fv_stats.png",
        dpi=200,
    )


if __name__ == "__main__":
    main()
