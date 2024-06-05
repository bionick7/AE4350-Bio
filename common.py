import pyray as rl


def color_from_html(html: int) -> rl.Color:
    return rl.Color((html >> 16) & 0xFF, (html >> 8) & 0xFF, html & 0xFF, 0xFF)

BG = color_from_html(0x2C2B30)
FG = color_from_html(0xEAEBDA)
HIGHLIGHT = color_from_html(0xE3524D)