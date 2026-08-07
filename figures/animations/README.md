# 3D animations

These animations are honest visualisations of the spectral (DFT) finding
from the paper. They are supplementary material — the static figures in
the parent `figures/` folder tell the same story in printed form.

## Files

| File | Word | Duration | What it shows |
|---|---|---|---|
| `spectral_waterfall_positive.mp4` | happy (V=8.47) | 6.0 s | DFT spectrum of "happy" growing letter by letter |
| `spectral_waterfall_negative.mp4` | torture (V=1.40) | 6.0 s | Same for "torture" — note the difference in low-frequency power |
| `word_trajectory_positive.mp4` | happy | 6.0 s | Word as a 3D walk through (a,b,c)-space |
| `word_trajectory_negative.mp4` | torture | 6.0 s | Same for "torture" |

## How to read them

**Spectral waterfall (Figure A):** x-axis is frequency bin (0 to 32), y-axis
is "letters added" (1 to 5 for "happy", 1 to 7 for "torture"), z-axis is
log-magnitude of the DFT. The camera rotates 360° over the 6 seconds.

**Word trajectory (Figure B):** each frame adds one more letter to the word.
The 3D point is (alphabet_pos[i], alphabet_pos[i+1], alphabet_pos[i+2]) —
consecutive letter triples. The trajectory is the line through these
triples. Letter positions: a=1, ..., z=26. The camera rotates while the
path builds.

## Regenerating

```bash
python -m src.animate
```

Requires `ffmpeg` on the system PATH for MP4 output. If ffmpeg is not
available, falls back to GIF via Pillow.
