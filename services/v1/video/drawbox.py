# Copyright (c) 2025 Stephen G. Pope
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.


import os
import subprocess
import logging
from services.file_management import download_file
from config import LOCAL_STORAGE_PATH, CUSTOM_FONTS_DIR

logger = logging.getLogger(__name__)


def _wrap_text(text, max_chars):
    """Split text into lines of at most max_chars characters, breaking on spaces where possible."""
    if not text or max_chars <= 0:
        return [text or ""]

    words = text.split(" ")
    lines = []
    current_line = ""

    for word in words:
        # A single word longer than max_chars: split by characters
        while len(word) > max_chars:
            if current_line:
                lines.append(current_line)
                current_line = ""
            lines.append(word[:max_chars])
            word = word[max_chars:]

        candidate = (current_line + " " + word).strip() if current_line else word
        if len(candidate) <= max_chars:
            current_line = candidate
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines if lines else [text]


def resolve_font_path(font_name):
    """
    Resolve a font name to an absolute path for FFmpeg fontfile=.
    Checks CUSTOM_FONTS_DIR first (filename without extension match), then fc-match.
    """
    if not font_name or not font_name.strip():
        raise ValueError("font is required")
    name = font_name.strip()
    if os.path.isabs(name) and os.path.isfile(name):
        return name
    if os.path.isdir(CUSTOM_FONTS_DIR):
        for f in os.listdir(CUSTOM_FONTS_DIR):
            if not f.lower().endswith(('.ttf', '.otf')):
                continue
            base, _ = os.path.splitext(f)
            if base.lower() == name.lower():
                return os.path.join(CUSTOM_FONTS_DIR, f)
    try:
        result = subprocess.run(
            ['fc-match', '-f', '%{file}', name],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout and os.path.isfile(result.stdout.strip()):
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    raise ValueError(f"Font '{font_name}' not found. Install the font or use a name from the custom fonts directory.")


def _drawbox_color_string(box_color, box_overlay):
    """Build drawbox color with optional alpha (e.g. black@0.8 or 0x000000@0.8)."""
    if not box_color:
        return "black"
    raw = box_color.strip()
    try:
        overlay = float(box_overlay)
    except (TypeError, ValueError):
        overlay = 1.0
    if overlay >= 1.0:
        return raw
    if raw.startswith('#'):
        hex_part = raw[1:]
        if len(hex_part) == 6:
            return f"0x{hex_part}@{overlay}"
    return f"{raw}@{overlay}"


def drawbox_video(
    video_url,
    box_x,
    box_y,
    box_w,
    box_h,
    text_x,
    text_y,
    text,
    font,
    job_id=None,
    box_color="black",
    box_overlay="1.0",
    box_t="fill",
    fontsize=24,
    fontcolor="white",
    text_align="center",
    text_max_width_margin=None,
    text_truncate_enabled=False,
):
    """
    Draw a box and text on a video using FFmpeg drawbox and drawtext filters.

    Returns:
        str: Path to the output video file.
    """
    import uuid
    if not job_id:
        job_id = str(uuid.uuid4())

    work_dir = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}_drawbox")
    os.makedirs(work_dir, exist_ok=True)

    input_path = download_file(video_url, work_dir)
    logger.info(f"Downloaded video to {input_path}")

    output_path = os.path.join(work_dir, f"{job_id}_drawbox.mp4")
    font_path = resolve_font_path(font).replace("'", "''")

    color_str = _drawbox_color_string(box_color, box_overlay)
    # drawbox: x, y, w, h, color, t (fill or thickness); allow expressions (e.g. h-150, iw)
    drawbox_part = f"drawbox=x={box_x}:y={box_y}:w={box_w}:h={box_h}:color={color_str}:t={box_t}"
    
    # Determine the list of text lines to render.
    # When text_truncate_enabled is set, wrap the text into multiple lines using Python
    # (FFmpeg drawtext has no reliable word-wrap; embedding \n in the text option causes
    # the backslash to be consumed by the filter-option parser, showing only 'n').
    # Each line gets its own drawtext filter with a calculated y offset instead.
    text_lines = [text or ""]  # default: single line
    if text_max_width_margin is not None and text_max_width_margin > 0 and text_truncate_enabled:
        try:
            probe_cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", input_path,
            ]
            import json as _json
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
            probe_data = _json.loads(probe_result.stdout)
            video_width = next(
                (s["width"] for s in probe_data.get("streams", []) if s.get("codec_type") == "video"),
                None,
            )
            if video_width:
                max_width_px = text_max_width_margin
                max_text_pixels = video_width - max_width_px
                char_width_estimate = fontsize * 0.9
                max_chars_per_line = max(1, int(max_text_pixels / char_width_estimate))
                text_lines = _wrap_text(text or "", max_chars_per_line)
        except Exception as probe_err:
            logger.warning(f"ffprobe failed for text wrapping: {probe_err}")

    # Build x position expression based on alignment and margin.
    # In drawtext expressions use 'w'/'h' (not 'iw'/'ih' which are unavailable there).
    # Wrap in single quotes so commas inside the expression are not treated as filter separators.
    if text_max_width_margin is not None and text_max_width_margin > 0:
        margin_per_side = text_max_width_margin // 2
        max_width_expr = f"w-{text_max_width_margin}"
        if text_align == "left":
            x_expr = f"'{margin_per_side}'"
        elif text_align == "right":
            x_expr = f"'w-text_w-{margin_per_side}'"
        else:  # center (default)
            x_expr = f"'lt(text_w,{max_width_expr})*(w-text_w)/2+gte(text_w,{max_width_expr})*{margin_per_side}'"
    else:
        x_expr = str(text_x)

    # Build one drawtext filter per line; offset each line's y by (fontsize + line_gap).
    line_gap = 4  # extra pixels between lines
    line_height = fontsize + line_gap
    drawtext_filters = []
    for i, line_text in enumerate(text_lines):
        line_esc = (line_text or "").replace("\\", "\\\\").replace("'", "''")
        y_val = str(text_y) if i == 0 else f"({text_y})+{i * line_height}"
        params = [
            f"text='{line_esc}'",
            f"fontfile='{font_path}'",
            f"fontsize={fontsize}",
            f"fontcolor={fontcolor}",
            f"x={x_expr}",
            f"y={y_val}",
        ]
        drawtext_filters.append(f"drawtext={':'.join(params)}")

    drawtext_part = ",".join(drawtext_filters)
    vf = f"{drawbox_part},{drawtext_part}"

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", vf,
        "-c:a", "copy",
        output_path,
    ]
    logger.info(f"Running FFmpeg: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"FFmpeg stderr: {result.stderr}")
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"FFmpeg failed: {result.stderr or result.stdout}")

    if os.path.exists(input_path):
        os.remove(input_path)
    return output_path
