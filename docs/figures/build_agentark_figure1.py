from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


HERE = Path(__file__).resolve().parent
DEFAULT_BASE = HERE / "agentark-figure1-base.png"
DEFAULT_OUTPUT = HERE / "agentark-figure1.png"


COLORS = {
    "ink": (30, 36, 47, 255),
    "muted": (76, 85, 100, 255),
    "blue": (39, 122, 213, 255),
    "green": (39, 145, 94, 255),
    "orange": (218, 133, 28, 255),
    "purple": (119, 80, 220, 255),
    "red": (202, 78, 74, 255),
    "white": (255, 255, 255, 216),
}


# Edit this section for paper/blog wording changes.
LABELS = [
    {
        "xy": (42, 26),
        "title": "1 Open-ended Env Scaling",
        "subtitle": "AI coding generates ever-growing task mods",
        "accent": "orange",
        "width": 640,
        "featured": True,
    },
    {
        "xy": (1020, 58),
        "title": "Multimodal Agent",
        "subtitle": "text + vision -> code action",
        "accent": "purple",
        "width": 420,
    },
    {
        "xy": (670, 410),
        "title": "AgentArk Shell",
        "subtitle": "loads mods and runs episodes",
        "accent": "blue",
        "width": 420,
    },
    {
        "xy": (52, 728),
        "title": "2 Multimodal Agent Eval",
        "subtitle": "benchmark via env interaction",
        "accent": "blue",
        "width": 520,
        "featured": True,
    },
    {
        "xy": (910, 690),
        "title": "3 Multimodal Agent RL",
        "subtitle": "train via repeated rollouts",
        "accent": "green",
        "width": 480,
        "featured": True,
    },
    {
        "xy": (512, 854),
        "title": "AgentArk Hub",
        "subtitle": "task pages and leaderboard",
        "accent": "red",
        "width": 410,
    },
]


PILLS = [
    {"xy": (74, 126), "text": "Coding Agent", "accent": "orange", "alpha": 210},
    {"xy": (380, 158), "text": "Task Mod Library", "accent": "orange", "alpha": 210},
    {"xy": (382, 512), "text": "feedback -> new task mods", "accent": "orange", "alpha": 184},
    {"xy": (724, 326), "text": "Code Executor", "accent": "purple"},
    {"xy": (1192, 224), "text": "Loaded Task Envs", "accent": "green"},
    {
        "xy": (602, 576),
        "text": "agent-ark Python Wrapper",
        "accent": "blue",
        "alpha": 190,
    },
]


POST_ARROW_PILLS = [
    {
        "xy": (360, 662),
        "text": "eval loop",
        "accent": "blue",
        "alpha": 174,
    },
    {
        "xy": (1030, 626),
        "text": "RL loop",
        "accent": "green",
        "alpha": 174,
    },
]


ARROWS = [
    {"points": [(316, 258), (360, 258)], "accent": "orange", "alpha": 175},
    {"points": [(584, 326), (632, 326)], "accent": "orange", "alpha": 175},
    {
        "points": [(1160, 250), (1075, 306), (912, 330)],
        "accent": "purple",
        "alpha": 145,
    },
    {
        "points": [(772, 570), (486, 612), (486, 728)],
        "accent": "blue",
        "alpha": 130,
    },
    {
        "points": [(1016, 535), (1206, 535), (1290, 590)],
        "accent": "green",
        "alpha": 130,
    },
    {
        "points": [(712, 258), (565, 166), (382, 142), (280, 172)],
        "accent": "orange",
        "alpha": 140,
        "width": 5,
        "head": 15,
    },
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Prefer the original Windows fonts, then common Linux fonts for regeneration.
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
        if bold
        else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def add_shadow(
    draw_image: Image.Image, rect: tuple[int, int, int, int], radius: int = 13
) -> None:
    x1, y1, x2, y2 = rect
    shadow = Image.new("RGBA", draw_image.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle(
        [x1 + 3, y1 + 5, x2 + 3, y2 + 5],
        radius=radius,
        fill=(0, 0, 0, 30),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(7))
    draw_image.alpha_composite(shadow)


def draw_label(
    draw_image: Image.Image,
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    title: str,
    subtitle: str,
    accent: tuple[int, int, int, int],
    width: int,
    featured: bool = False,
) -> None:
    title_font = font(34 if featured else 28, True)
    sub_font = font(21 if featured else 18, False)
    x, y = xy
    title_box = draw.textbbox((0, 0), title, font=title_font)
    sub_box = draw.textbbox((0, 0), subtitle, font=sub_font)
    pad_x, pad_y = (18, 13) if featured else (14, 10)
    height = (
        title_box[3]
        - title_box[1]
        + sub_box[3]
        - sub_box[1]
        + 6
        + pad_y * 2
    )
    rect = (x, y, x + width, y + height)
    add_shadow(draw_image, rect)
    fill_alpha = 188 if featured else COLORS["white"][3]
    outline = (
        (accent[0], accent[1], accent[2], 190)
        if featured
        else (255, 255, 255, 238)
    )
    draw.rounded_rectangle(
        rect,
        radius=13,
        fill=(255, 255, 255, fill_alpha),
        outline=outline,
        width=3 if featured else 2,
    )
    bar_width = 14 if featured else 9
    draw.rounded_rectangle([x, y, x + bar_width, y + height], radius=13, fill=accent)
    text_x = x + pad_x + 5
    text_y = y + pad_y - 2
    draw.text(
        (text_x, text_y),
        title,
        fill=accent if featured else COLORS["ink"],
        font=title_font,
    )
    draw.text(
        (text_x, text_y + title_box[3] - title_box[1] + 8),
        subtitle,
        fill=(45, 52, 65, 255) if featured else COLORS["muted"],
        font=sub_font,
    )


def draw_pill(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    accent: tuple[int, int, int, int],
    alpha: int = 202,
) -> None:
    small_font = font(18, False)
    x, y = xy
    text_box = draw.textbbox((0, 0), text, font=small_font)
    width = text_box[2] - text_box[0] + 20
    height = text_box[3] - text_box[1] + 12
    draw.rounded_rectangle(
        [x, y, x + width, y + height],
        radius=12,
        fill=(255, 255, 255, alpha),
        outline=accent,
        width=2,
    )
    draw.text((x + 10, y + 6), text, fill=accent, font=small_font)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    color: tuple[int, int, int, int],
    width: int = 4,
    head: int = 13,
) -> None:
    for start, end in zip(points, points[1:]):
        draw.line([start, end], fill=color, width=width, joint="curve")
    (x1, y1), (x2, y2) = points[-2], points[-1]
    angle = math.atan2(y2 - y1, x2 - x1)
    p1 = (
        x2 - head * math.cos(angle - math.pi / 6),
        y2 - head * math.sin(angle - math.pi / 6),
    )
    p2 = (
        x2 - head * math.cos(angle + math.pi / 6),
        y2 - head * math.sin(angle + math.pi / 6),
    )
    draw.polygon([(x2, y2), p1, p2], fill=color)


def build_figure(base_path: Path, output_path: Path) -> None:
    image = Image.open(base_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for item in ARROWS:
        base = COLORS[item["accent"]]
        color = (base[0], base[1], base[2], item["alpha"])
        draw_arrow(
            draw,
            item["points"],
            color,
            width=item.get("width", 4),
            head=item.get("head", 13),
        )

    for item in LABELS:
        draw_label(
            overlay,
            draw,
            item["xy"],
            item["title"],
            item["subtitle"],
            COLORS[item["accent"]],
            item["width"],
            item.get("featured", False),
        )

    for item in PILLS:
        draw_pill(
            draw,
            item["xy"],
            item["text"],
            COLORS[item["accent"]],
            item.get("alpha", 202),
        )

    for item in POST_ARROW_PILLS:
        draw_pill(
            draw,
            item["xy"],
            item["text"],
            COLORS[item["accent"]],
            item.get("alpha", 202),
        )

    result = Image.alpha_composite(image, overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, quality=96)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the AgentArk Figure 1 image.")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_figure(args.base, args.out)
