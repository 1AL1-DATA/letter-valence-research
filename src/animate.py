"""
3D animations for the letter-valence research paper.

Two figures, both honest visualisations of the spectral finding:

  Figure A — Spectral waterfall
    For a single word, show its DFT spectrum evolving as the word grows
    letter by letter. x = frequency bin, y = letters added (1..N), z = log
    magnitude. The animation shows that the spectrum at low frequencies
    differs between positive-valence and negative-valence words.

  Figure B — Word as 3D trajectory
    For a single word, plot the alphabet positions of consecutive
    letter triples as a 3D walk. Each frame adds one more point; the
    trajectory is the path the word traces through (a,b,c)-space
    where a, b, c are consecutive alphabet positions. The current
    letter is highlighted.

Outputs:
  figures/animations/spectral_waterfall_positive.mp4
  figures/animations/spectral_waterfall_negative.mp4
  figures/animations/word_trajectory_positive.mp4
  figures/animations/word_trajectory_negative.mp4
  figures/animations/word_trajectory_positive.gif   (fallback if ffmpeg OK)
  figures/animations/word_trajectory_negative.gif

Run:
  python -m src.animate
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.style import apply_style, SEMANTIC
from src.visualise import dft_spectrum

apply_style()

OUT_DIR = REPO / "figures" / "animations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Exemplar words (5-8 letters, real valence extremes from Warriner 2013)
WORDS = {
    "positive": "happy",     # V = 8.47, 5 letters
    "negative": "torture",   # V = 1.40, 7 letters
}

# Animation parameters
FPS = 24
N_BINS = 32       # DFT bin count
ROTATE_PERIOD = 6.0   # seconds for one full 360° rotation


# ══════════════════════════════════════════════════════════════════════════════
# Figure A — Spectral waterfall
# ══════════════════════════════════════════════════════════════════════════════

def make_spectral_waterfall(word: str, out_path: Path) -> None:
    """
    For each prefix length p in 1..len(word), compute the DFT spectrum
    of the prefix. Stack these as rows in a 2D array. Plot as a 3D
    waterfall where x = frequency bin, y = prefix length, z = log
    magnitude. The animation rotates the camera.
    """
    from matplotlib import animation

    word = word.lower()
    n_total = len(word)

    # Pre-compute spectra for each prefix
    spectra = []
    for p in range(1, n_total + 1):
        prefix = word[:p]
        mag, _ = dft_spectrum(prefix, n_bins=N_BINS)
        spectra.append(mag)
    Z = np.array(spectra)  # shape: (n_total, n_bins)
    # z is already log-magnitude from dft_spectrum, but ensure it's a real
    # finite array for plotting
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    x = np.arange(N_BINS)
    y = np.arange(1, n_total + 1)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Pick a colormap that goes dark → bright along the spectrum magnitude
    cmap = plt.get_cmap("magma")
    norm = plt.Normalize(vmin=Z.min(), vmax=Z.max())
    facecolors = cmap(norm(Z))

    # Plot as a 3D bar chart where each bar's height = spectrum magnitude.
    # The bar chart makes the prefix-accumulation visible.
    bar_width = 0.7
    for i in range(n_total):
        for j in range(N_BINS):
            ax.bar3d(
                j, i, 0,
                bar_width, 0.8, Z[i, j],
                color=cmap(norm(Z[i, j])),
                edgecolor="none",
                alpha=0.9,
            )

    ax.set_xlabel("Frequency bin")
    ax.set_ylabel("Letters added")
    ax.set_zlabel("log |DFT|")
    ax.set_title(
        f"Spectral waterfall — word grows: '{word}'\n"
        f"Each row = DFT of the prefix (first i letters)"
    )
    ax.set_xlim(0, N_BINS)
    ax.set_ylim(0, n_total + 1)
    ax.set_zlim(0, Z.max() * 1.1)
    ax.view_init(elev=25, azim=-60)

    def rotate(frame: int):
        # Full 360° rotation over FPS * ROTATE_PERIOD frames
        total_frames = int(FPS * ROTATE_PERIOD)
        azim = -60 + 360 * (frame / total_frames)
        ax.view_init(elev=25, azim=azim)
        return []

    ani = animation.FuncAnimation(
        fig, rotate, frames=int(FPS * ROTATE_PERIOD),
        interval=1000 / FPS, blit=False,
    )

    # Try MP4 first, fall back to GIF
    mp4_path = out_path.with_suffix(".mp4")
    try:
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=FPS, metadata=dict(artist="letter-valence"),
                        bitrate=2400)
        ani.save(str(mp4_path), writer=writer)
        print(f"  Wrote {mp4_path.name}")
    except (KeyError, RuntimeError) as e:
        print(f"  ffmpeg writer failed ({e!r}); falling back to GIF")
        gif_path = out_path.with_suffix(".gif")
        ani.save(str(gif_path), writer="pillow", fps=FPS)
        print(f"  Wrote {gif_path.name}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure B — Word as 3D trajectory
# ══════════════════════════════════════════════════════════════════════════════

def make_word_trajectory(word: str, out_path: Path) -> None:
    """
    Plot a word as a 3D walk where each frame adds one more letter.
    The 3D point at step k is (alphabet_pos[k], alphabet_pos[k+1],
    alphabet_pos[k+2]) — consecutive letter triples. The trajectory
    is the line through these triples. The current letter is highlighted.

    The animation uses a FIXED total frame count (FPS * ROTATE_PERIOD) so
    the camera rotation plays out over the full 6 seconds. Between letter
    additions, the current point interpolates smoothly so the path looks
    continuous rather than jumpy.
    """
    from matplotlib import animation

    word = word.lower()
    positions = [ord(c) - ord("a") + 1 for c in word if c.isalpha()]
    if len(positions) < 3:
        print(f"  Skipping '{word}' — need at least 3 letters for triple")
        return
    # Pad with 0s at the start so the first two frames are (0, a, b) and
    # (a, b, c) — this means the "letter being added" maps cleanly:
    # frame 0 → "start" (no letter yet), frame 1 → "first letter", etc.
    padded = [0.0] * 2 + [float(p) for p in positions]
    n_steps = len(padded) - 2           # number of triples
    total_frames = int(FPS * ROTATE_PERIOD)  # total animation length

    # Pre-compute all triples for the path
    triples = np.array(
        [padded[i:i + 3] for i in range(n_steps)],
        dtype=float,
    )  # shape: (n_steps, 3)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    accent = SEMANTIC["secondary"]
    text_c = SEMANTIC["text"]

    # Static scatter of all triples (small dots, faded)
    ax.scatter(
        triples[:, 0], triples[:, 1], triples[:, 2],
        c=accent, s=30, alpha=0.35, depthshade=True,
    )

    # Animated line and current-letter marker
    line, = ax.plot([], [], [], color=accent, linewidth=2.0, alpha=0.85)
    current_dot = ax.scatter(
        [], [], [], c=accent, s=180, edgecolors=text_c, linewidths=1.5,
        depthshade=False,
    )
    label = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=14)

    ax.set_xlabel("letter[i]")
    ax.set_ylabel("letter[i+1]")
    ax.set_zlabel("letter[i+2]")
    ax.set_xlim(0, 27)
    ax.set_ylim(0, 27)
    ax.set_zlim(0, 27)
    ax.set_title(
        f"Word as a 3D walk — '{word}'\n"
        f"Each step: (letter[i], letter[i+1], letter[i+2])  |  a=1, ..., z=26"
    )
    ax.view_init(elev=22, azim=-55)

    def interpolate_point(t_norm: float) -> tuple[int, np.ndarray]:
        """
        Map a normalised time t_norm in [0, 1] to a path position.
        The path advances linearly through the triples, and we
        interpolate between consecutive triples so the line is smooth.
        """
        # Float index into the path
        fidx = t_norm * (n_steps - 1)
        i = min(int(fidx), n_steps - 2)
        frac = fidx - i
        # Linear interpolation between triples[i] and triples[i+1]
        p = triples[i] * (1 - frac) + triples[i + 1] * frac
        return i, p

    def update(frame: int):
        t_norm = frame / max(total_frames - 1, 1)
        # The path: the line goes from triples[0] to the current point
        _, current = interpolate_point(t_norm)
        # Show the full path up to the current step (with the moving tip)
        path = np.vstack([triples[0:1], current[None, :]]) if t_norm < 1e-3 \
            else np.vstack([triples[: max(1, int(t_norm * (n_steps - 1)) + 1)],
                            current[None, :]])
        # Simpler: show the full up-to-current path, then add the moving tip
        step_idx = int(t_norm * (n_steps - 1))
        path_so_far = triples[: step_idx + 1].copy()
        if step_idx < n_steps - 1:
            path_so_far = np.vstack([path_so_far, current[None, :]])
        line.set_data(path_so_far[:, 0], path_so_far[:, 1])
        line.set_3d_properties(path_so_far[:, 2])
        current_dot._offsets3d = (
            [current[0]], [current[1]], [current[2]]
        )
        # Label: which letter is being added at this step
        if step_idx < 2:
            label_text = "start"
        else:
            letter_idx = step_idx - 2
            if letter_idx < len(word):
                label_text = f"step {letter_idx + 1}: '{word[letter_idx]}'"
            else:
                label_text = f"step {letter_idx + 1}"
        label.set_text(label_text)
        # Camera rotates linearly over the full animation
        azim = -55 + 360 * t_norm
        ax.view_init(elev=22, azim=azim)
        return line, current_dot, label

    ani = animation.FuncAnimation(
        fig, update, frames=total_frames,
        interval=1000 / FPS, blit=False,
    )

    mp4_path = out_path.with_suffix(".mp4")
    try:
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=FPS, metadata=dict(artist="letter-valence"),
                        bitrate=2400)
        ani.save(str(mp4_path), writer=writer)
        print(f"  Wrote {mp4_path.name}")
    except (KeyError, RuntimeError) as e:
        print(f"  ffmpeg writer failed ({e!r}); falling back to GIF")
        gif_path = out_path.with_suffix(".gif")
        ani.save(str(gif_path), writer="pillow", fps=FPS)
        print(f"  Wrote {gif_path.name}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print("ANIMATIONS: spectral waterfall + 3D word trajectory")
    print("=" * 70)
    print()
    for label, word in WORDS.items():
        print(f"[{label}] word: '{word}'")
        print(f"  -> Figure A: spectral waterfall")
        make_spectral_waterfall(word, OUT_DIR / f"spectral_waterfall_{label}")
        print(f"  -> Figure B: 3D trajectory")
        make_word_trajectory(word, OUT_DIR / f"word_trajectory_{label}")
        print()
    print("=" * 70)
    print(f"Done. Outputs in {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
