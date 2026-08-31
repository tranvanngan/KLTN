from __future__ import annotations

import matplotlib


def set_time_new_roman_font() -> None:
    """Set global matplotlib font to Times New Roman (paper-friendly).

    Notes:
    - On Windows, matplotlib typically finds Times New Roman if installed.
    - We also set mathtext to Times-like glyphs.
    """

    # Font family
    matplotlib.rcParams["font.family"] = "Times New Roman"
    # Fallbacks in case the exact font name isn't available
    matplotlib.rcParams["font.sans-serif"] = ["Times New Roman", "DejaVu Sans"]
    matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

    # Math rendering
    matplotlib.rcParams["mathtext.fontset"] = "stix"

    # Avoid minus sign being rendered as a square box
    matplotlib.rcParams["axes.unicode_minus"] = False

