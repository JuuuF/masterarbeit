import os
import json
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import defaultdict

out_dir = "writing/work/imgs/rendering/ergebnisse"

print("Loading Data...", flush=True)
src_dir = "data/generation/out"
dirs = [os.path.join(src_dir, d) for d in os.listdir(src_dir)]

# dirs = dirs[:1000]

infos = [os.path.join(d, "info.json") for d in dirs]
infos = [pd.Series(json.load(open(f, "r"))) for f in infos]

print("Gathering Information...", flush=True)
df = pd.DataFrame(infos)
n_samples = df.shape[0]

# Camera
cam_angles = df.loc[:, "camera_angle"]
cam_apertures = df.loc[:, "camera_aperture"]
cam_dists = df.loc[:, "camera_distance"]
cam_exposures = df.loc[:, "camera_exposure"]
cam_focal = df.loc[:, "camera_focal_mm"]
cam_focus_dist = df.loc[:, "camera_focus_distance"]
cam_motion_blur = df.loc[:, "camera_motion_blur"]

# Darts
dart_count = df.loc[:, "dart_count"]
dart_dists = df.loc[:, "dart_distances"]

# Lights
cabinet = df.loc[:, "cabinet"]
light_flash = df.loc[:, "light_camera_flash"]
light_ceiling = df.loc[:, "light_ceiling"]
light_ring = df.loc[:, "light_ring"]
light_spot = df.loc[:, "light_spot"]
env_texture_strength = df.loc[:, "env_texture_strength"]

# Scores
scores = df.loc[:, "scores"]
scores = [
    s[1] if s[1] != "OUT" else 0 for scr in scores for s in scr if s[1] != "HIDDEN"
]
total_darts = len(scores)
print(df.columns)

# ------------------------------------------------------------------------


print("Plotting Data...", flush=True)


def violin(data, title, filename):
    # Create a figure and an axis object.
    fig, ax = plt.subplots()
    ax.tick_params(axis="both", which="major", labelsize=14)

    # Create the violin plot.
    # showmeans=True and showmedians=True add markers for the mean and median, respectively.
    vp = ax.violinplot(data, showmeans=True, showmedians=False)
    for body in vp["bodies"]:
        body.set_facecolor("#AA95BD")
        body.set_edgecolor("black")
        body.set_alpha(0.7)

    # jitter = np.random.uniform(-0.125, 0.125, size=len(data)) + 1
    # ax.scatter(jitter, data, color="#76608A", alpha=0.1, s=1)

    # Clear ticks
    ax.set_xticks([])

    # Add labels and title.
    # ax.set_title(title)

    fig.patch.set_facecolor("none")
    ax.patch.set_facecolor("none")

    print("Saving Plot", filename, flush=True)
    # Display the plot.
    plt.savefig(f"{out_dir}/{filename}.pdf", transparent=True)


os.makedirs(f"{out_dir}", exist_ok=True)
violin(cam_angles.values, "Kamerawinkel zur Dartscheibe [°]", "cam_angles")
# violin(cam_apertures.values, "Öffnung der Kamera [mm]", "cam_apertures")
violin(cam_dists.values, "Kameraabstand zur Dartscheibe [m]", "cam_dists")
# violin(cam_exposures.values, "Kamerabelichtung", "cam_exposures")
violin(cam_focal.values, "Brennweite der Kamera [mm]", "cam_focal")
# violin(cam_focus_dist.values, "Kamerafokus [mm]", "cam_focus_dist")
# violin(cam_motion_blur.values, "Bewegungsunschärfe [cm]", "cam_motion_blur")
violin(
    env_texture_strength.values,
    "Helligkeit der Umgebungstextur",
    "env_texture_strength",
)


def bar(data, title, filename):
    # Calculate frequency counts for each category (0, 1, 2, 3)
    categories = [0, 1, 2, 3]
    counts = [np.sum(data == cat) for cat in categories]

    # Compute relative frequencies (as fractions)
    total = sum(counts)
    rel_freq = [100 * count / total for count in counts]

    # Create a figure and an axes object.
    fig, ax = plt.subplots()
    ax.tick_params(axis="both", which="major", labelsize=14)

    # Set bar properties:
    # - Bar face color: light purple (#AA95BD)
    # - Bar edge color: black
    # - Bar transparency: fully opaque here (you can adjust the alpha if desired)
    bar_width = 0.5
    x = np.arange(len(categories))  # positions [0,1,2,3] on the x-axis
    bars = ax.bar(x, rel_freq, width=bar_width, color="#AA95BD", edgecolor="black")

    # Customize the x-axis: set ticks and labels for each category.
    ax.set_xticks(x)
    ax.set_xticklabels([str(cat) for cat in categories])
    # ax.set_title(title)
    ax.set_ylim((0, 100))

    for i, total in enumerate(rel_freq):
        ax.text(
            i,
            total,
            f"{total:.02f}%",
            ha="center",
            va="bottom",
            fontsize=14,
        )

    # (Optional) Set both the figure and the axes backgrounds to transparent.
    fig.patch.set_facecolor("none")
    ax.patch.set_facecolor("none")

    print("Saving Plot", filename, flush=True)
    # Display the plot.
    plt.savefig(f"{out_dir}/{filename}.pdf", transparent=True)


bar(dart_count.values, "Anzahl Dartpfeile je Bild", "dart_counts")


def lights():
    perc_flash = light_flash.mean() * 100
    perc_ceiling = light_ceiling.mean() * 100
    perc_ring = light_ring.mean() * 100
    perc_spot = light_spot.mean() * 100
    perc_cabinet = cabinet.mean() * 100

    # Define the categories and their corresponding percentages.
    categories = ["Deckenlicht", "Blitz", "Spotlight", "Ringlicht", "Schrank"]
    percentages = [perc_ceiling, perc_flash, perc_spot, perc_ring, perc_cabinet]

    # Create the bar chart.
    fig, ax = plt.subplots()
    ax.tick_params(axis="both", which="major", labelsize=14)

    # Setup bar properties: light purple (#AA95BD) face color and black edge.
    bar_width = 0.5
    x_positions = np.arange(len(categories))
    bars = ax.bar(
        x_positions, percentages, width=bar_width, color="#AA95BD", edgecolor="black"
    )

    # Customize the axes.
    ax.set_xticks(x_positions)
    ax.set_xticklabels(categories)
    # ax.set_title("Lichter in den Daten")
    ax.set_ylim((0, 100))

    for i, total in enumerate(percentages):
        ax.text(
            i,
            total,
            f"{total:.02f}%",
            ha="center",
            va="bottom",
            fontsize=14,
        )

    # Set both figure and axis backgrounds to transparent.
    fig.patch.set_facecolor("none")
    ax.patch.set_facecolor("none")

    print("Saving Plot lights_bar_chart", flush=True)

    # Save the figure as a transparent PDF.
    plt.savefig(f"{out_dir}/lights_bar_chart.pdf", transparent=True, format="pdf")


lights()


def plot_scores_dart():
    color_singles = "#A6CEE3"  # Black for singles
    color_doubles = "#1F78B4"  # Red for doubles
    color_triples = "#B2DF8A"  # Green for triples

    # We want to count occurrences for each category in three modes:
    # [singles, doubles, triples]
    # Interpretation:
    # - A plain value (or "B") is a single.
    # - A value that begins with "D" is counted as a double.
    # - A value that begins with "T" is counted as a triple.
    counts = defaultdict(lambda: [0, 0, 0])
    for entry in scores:
        if isinstance(entry, int):
            key = str(entry)
            counts[key][0] += 1
        elif isinstance(entry, str):
            if entry == "B":
                counts["B"][0] += 1
            elif entry.startswith("D"):
                key = entry[1:]
                counts[key][1] += 1
            elif entry.startswith("T"):
                key = entry[1:]
                counts[key][2] += 1
            else:
                key = entry
                counts[key][0] += 1

    # -------------------------------
    # Define the dartboard ordering (for the ring)
    # Standard dartboard order (clockwise, starting from the top):
    dartboard_order = [
        "10",
        "15",
        "2",
        "17",
        "3",
        "19",
        "7",
        "16",
        "8",
        "11",
        "14",
        "9",
        "12",
        "5",
        "20",
        "1",
        "18",
        "4",
        "13",
        "6",
    ][::-1]

    # These will form our polar (ring) chart categories.
    ring_categories = dartboard_order

    # For the bullseye and 0, we define:
    bull0_categories = ["B", "0"]

    # Ensure every category exists
    for cat in ring_categories + bull0_categories:
        if cat not in counts:
            counts[cat] = [0, 0, 0]

    # Extract stacked data (singles, doubles, triples) for the ring categories:
    ring_singles = np.array([counts[cat][0] for cat in ring_categories]) / total_darts
    ring_doubles = np.array([counts[cat][1] for cat in ring_categories]) / total_darts
    ring_triples = np.array([counts[cat][2] for cat in ring_categories]) / total_darts

    # And for the bull/0 categories:
    bull0_singles = np.array([counts[cat][0] for cat in bull0_categories]) / total_darts
    bull0_doubles = np.array([counts[cat][1] for cat in bull0_categories]) / total_darts
    bull0_triples = np.array([counts[cat][2] for cat in bull0_categories]) / total_darts

    # -------------------------------
    # Create the figure with two subplots:
    # 1. A polar subplot (for the dartboard ring)
    # 2. A normal subplot (for bull and 0) placed side by side

    fig = plt.figure(figsize=(12, 6))
    ax_polar = fig.add_subplot(1, 2, 1, projection="polar")
    ax_bull0 = fig.add_subplot(1, 2, 2)
    ax_polar.tick_params(axis="both", which="major", labelsize=16)
    ax_bull0.tick_params(axis="both", which="major", labelsize=14)

    # --- Polar (ring) chart ---
    N = len(ring_categories)
    # Evenly spaced angles for each segment:
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # Set the angular width of each bar (80% of the available space per segment)
    width = 2 * np.pi / N * 0.8

    # Plot the singles (base layer)
    bars_singles = ax_polar.bar(
        theta,
        ring_singles,
        width=width,
        bottom=0,
        color=color_singles,
        edgecolor="black",
        label="Singles",
    )
    # Plot doubles on top
    bars_doubles = ax_polar.bar(
        theta,
        ring_doubles,
        width=width,
        bottom=ring_singles,
        color=color_doubles,
        edgecolor="black",
        label="Doubles",
    )
    # Plot triples on top (stack on singles+doubles)
    bottom_for_triples = ring_singles + ring_doubles
    bars_triples = ax_polar.bar(
        theta,
        ring_triples,
        width=width,
        bottom=bottom_for_triples,
        color=color_triples,
        edgecolor="black",
        label="Triples",
    )

    # Set the theta ticks to our dartboard numbers.
    ax_polar.set_xticks(theta)
    ax_polar.set_xticklabels(ring_categories)
    ax_polar.set_title("Felder 1-20", va="bottom", fontsize=16)
    ax_polar.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1), fontsize=14)
    ax_polar.set_yticklabels([])
    ymax = max(ring_singles + ring_doubles + ring_triples) * 1.2
    ax_polar.set_ylim((0, ymax))

    for i, cat in enumerate(ring_categories):
        total = ring_singles[i] + ring_doubles[i] + ring_triples[i]
        ax_polar.text(
            theta[i],
            (total + ymax) / 2,
            f"{total*100:.02f}%",
            ha="center",
            va="center",
            fontsize=12,
        )

    # --- Bull/0 chart ---
    # For simplicity, we create a vertical stacked bar chart.
    x = np.arange(len(bull0_categories)) / 2  # positions for the two categories
    bars_b0_singles = ax_bull0.bar(
        x,
        bull0_singles,
        color=color_singles,
        edgecolor="black",
        label="Singles",
        width=0.2,
    )
    bars_b0_doubles = ax_bull0.bar(
        x,
        bull0_doubles,
        bottom=bull0_singles,
        color=color_doubles,
        edgecolor="black",
        label="Doubles",
        width=0.2,
    )
    bars_b0_triples = ax_bull0.bar(
        x,
        bull0_triples,
        bottom=bull0_singles + bull0_doubles,
        color=color_triples,
        edgecolor="black",
        label="Triples",
        width=0.2,
    )

    ax_bull0.set_xticks(x)
    ax_bull0.set_xticklabels(bull0_categories)
    ax_bull0.set_title("Bull & Out", va="bottom", fontsize=16)
    ax_bull0.set_ylim((0, 0.25))

    # Optionally, add text on top of each bar showing total count.
    for i, cat in enumerate(bull0_categories):
        total = bull0_singles[i] + bull0_doubles[i] + bull0_triples[i]
        ax_bull0.text(
            x[i],
            total,
            f"{total*100:.02f}%",
            ha="center",
            va="bottom",
            fontsize=14,
        )

    # -------------------------------
    # Set transparent backgrounds:
    fig.patch.set_facecolor("none")
    ax_polar.patch.set_facecolor("none")
    ax_bull0.patch.set_facecolor("none")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/dartboard_stacked.pdf", transparent=True, format="pdf")


plot_scores_dart()
