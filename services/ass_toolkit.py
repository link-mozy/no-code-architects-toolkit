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
import ffmpeg
import logging
import subprocess
import whisper
from datetime import timedelta
import srt
import re
from services.file_management import download_file
from services.cloud_storage import upload_file  # Ensure this import is present
import requests  # Ensure requests is imported for webhook handling
from urllib.parse import urlparse
from config import LOCAL_STORAGE_PATH, CUSTOM_FONTS_DIR

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

POSITION_ALIGNMENT_MAP = {
    "bottom_left": 1,
    "bottom_center": 2,
    "bottom_right": 3,
    "middle_left": 4,
    "middle_center": 5,
    "middle_right": 6,
    "top_left": 7,
    "top_center": 8,
    "top_right": 9
}

def rgb_to_ass_color(rgb_color):
    """Convert RGB hex to ASS (&HAABBGGRR)."""
    if isinstance(rgb_color, str):
        rgb_color = rgb_color.lstrip('#')
        if len(rgb_color) == 6:
            r = int(rgb_color[0:2], 16)
            g = int(rgb_color[2:4], 16)
            b = int(rgb_color[4:6], 16)
            return f"&H00{b:02X}{g:02X}{r:02X}"
    return "&H00FFFFFF"

def generate_transcription(video_path, language='auto'):
    try:
        model = whisper.load_model("base")
        transcription_options = {
            'word_timestamps': True,
            'verbose': True,
        }
        if language != 'auto':
            transcription_options['language'] = language
        result = model.transcribe(video_path, **transcription_options)
        logger.info(f"Transcription generated successfully for video: {video_path}")
        return result
    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        raise

def get_video_resolution(video_path):
    try:
        probe = ffmpeg.probe(video_path)
        video_streams = [s for s in probe['streams'] if s['codec_type'] == 'video']
        if video_streams:
            width = int(video_streams[0]['width'])
            height = int(video_streams[0]['height'])
            logger.info(f"Video resolution determined: {width}x{height}")
            return width, height
        else:
            logger.warning(f"No video streams found for {video_path}. Using default resolution 384x288.")
            return 384, 288
    except Exception as e:
        logger.error(f"Error getting video resolution: {str(e)}. Using default resolution 384x288.")
        return 384, 288

# All font filenames (without extension) in CUSTOM_FONTS_DIR are accepted as font_family.
# This set is only for ASS output: these names have fontconfig aliases, so we use them as-is
# in the ASS file instead of mapping to fc-query family (which would lose Bold/Italic).
FONTCONFIG_ALIAS_NAMES = {'arialbd', 'ariali', 'arialbi'}

_custom_fonts_cache = None  # (set of filename_no_ext, dict filename_no_ext -> family for ASS)


def _get_custom_fonts_cache():
    """Scan CUSTOM_FONTS_DIR. Returns (set of all accepted font names = filenames without extension, filename_no_ext -> family for ASS)."""
    global _custom_fonts_cache
    if _custom_fonts_cache is not None:
        return _custom_fonts_cache
    names = set()
    filename_to_family = {}
    if not os.path.isdir(CUSTOM_FONTS_DIR):
        _custom_fonts_cache = (names, filename_to_family)
        return _custom_fonts_cache
    for f in os.listdir(CUSTOM_FONTS_DIR):
        if not f.lower().endswith(('.ttf', '.otf')):
            continue
        path = os.path.join(CUSTOM_FONTS_DIR, f)
        if not os.path.isfile(path):
            continue
        name_no_ext = os.path.splitext(f)[0]
        names.add(name_no_ext)  # every font filename (no ext) is accepted as font_family
        if name_no_ext.lower() in FONTCONFIG_ALIAS_NAMES:
            continue  # skip mapping for these; they use fontconfig alias in ASS
        try:
            r = subprocess.run(
                ['fc-query', '--format=%{family}\n', path],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if r.returncode == 0 and r.stdout and r.stdout.strip():
                # ASS uses comma as field separator; use only first family if fc-query returns "Name1, Name2"
                family = r.stdout.strip().split('\n')[0].strip().split(',')[0].strip()
                if family:
                    filename_to_family[name_no_ext] = family
        except (subprocess.TimeoutExpired, OSError, Exception):
            pass
    _custom_fonts_cache = (names, filename_to_family)
    logger.info(f"Custom fonts: {len(names)} names, {len(filename_to_family)} filename->family mappings from {CUSTOM_FONTS_DIR}")
    return _custom_fonts_cache


def resolve_font_family_for_ass(font_family):
    """Return the font name to use in the ASS file (fontconfig-resolvable). Uses filename->family map when needed.
    ASS style line uses comma as separator; returned name must not contain commas."""
    _, filename_to_family = _get_custom_fonts_cache()
    for name_no_ext, family in filename_to_family.items():
        if name_no_ext.lower() == font_family.lower():
            # ASS forbids comma in Fontname; take first family only
            return family.split(',')[0].strip() if family else font_family
    # Ensure no comma in output (ASS uses comma as field separator)
    return (font_family or '').split(',')[0].strip() or font_family


def get_available_fonts():
    """Get the list of available fonts: fontconfig (fc-list) + custom fonts dir (filename without extension)."""
    font_names = set()
    try:
        result = subprocess.run(
            ['fc-list', ':', 'family'],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.splitlines():
            families = line.split(',')
            for family in families:
                family_name = family.strip()
                if family_name:
                    font_names.add(family_name)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.debug(f"fc-list not used: {e}")
    custom_names, _ = _get_custom_fonts_cache()
    font_names.update(custom_names)
    font_names_lower = {f.lower() for f in font_names}
    if 'arial' in font_names_lower:
        font_names.update(('ARIALBD', 'ARIALI', 'ARIALBI'))
    logger.info(f"Available fonts: {len(font_names)} (fontconfig + custom dir)")
    return list(font_names)

def format_ass_time(seconds):
    """Convert float seconds to ASS time format H:MM:SS.cc"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centiseconds = int(round((seconds - int(seconds)) * 100))
    return f"{hours}:{minutes:02}:{secs:02}.{centiseconds:02}"

def process_subtitle_text(text, replace_dict, all_caps, max_words_per_line):
    """Apply text transformations: replacements, all caps, and optional line splitting."""
    for old_word, new_word in replace_dict.items():
        text = re.sub(re.escape(old_word), new_word, text, flags=re.IGNORECASE)
    if all_caps:
        text = text.upper()
    if max_words_per_line > 0:
        words = text.split()
        lines = [' '.join(words[i:i+max_words_per_line]) for i in range(0, len(words), max_words_per_line)]
        text = '\\N'.join(lines)
    return text

def is_srt_format(content):
    """
    Check if content is in SRT format.
    SRT format typically starts with a number, followed by timestamp lines.
    """
    if not content or not isinstance(content, str):
        return False
    
    content = content.strip()
    if not content:
        return False
    
    # Try to parse as SRT - if it succeeds, it's SRT format
    try:
        subtitles = list(srt.parse(content))
        # If parsing succeeds and we got at least one subtitle, it's SRT
        return len(subtitles) > 0
    except (srt.SRTParseError, ValueError, AttributeError):
        # If parsing fails, it's not SRT format
        return False

def get_video_duration(video_path):
    """Get video duration in seconds using ffprobe."""
    try:
        probe_cmd = [
            'ffprobe', 
            '-v', 'error', 
            '-show_entries', 'format=duration', 
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        duration_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        duration = float(duration_result.stdout.strip())
        return duration
    except (ValueError, subprocess.SubprocessError) as e:
        logger.warning(f"Could not determine video duration: {str(e)}")
        return None

def plain_text_to_transcription_result(text, video_duration):
    """
    Convert plain text into a transcription-like structure.
    The text will be displayed for the entire video duration.
    """
    if video_duration is None or video_duration <= 0:
        # Fallback: use 10 seconds if duration cannot be determined
        video_duration = 10.0
        logger.warning("Video duration not available, using 10 seconds as fallback.")
    
    segments = [{
        'start': 0.0,
        'end': video_duration,
        'text': text.strip(),
        'words': []
    }]
    logger.info(f"Converted plain text to transcription result (duration: {video_duration}s).")
    return {'segments': segments}

def srt_to_transcription_result(srt_content):
    """Convert SRT content into a transcription-like structure for uniform processing."""
    subtitles = list(srt.parse(srt_content))
    segments = []
    for sub in subtitles:
        segments.append({
            'start': sub.start.total_seconds(),
            'end': sub.end.total_seconds(),
            'text': sub.content.strip(),
            'words': []  # SRT does not provide word-level timestamps
        })
    logger.info("Converted SRT content to transcription result.")
    return {'segments': segments}

def split_lines(text, max_words_per_line):
    """Split text into multiple lines if max_words_per_line > 0."""
    if max_words_per_line <= 0:
        return [text]
    words = text.split()
    lines = [' '.join(words[i:i+max_words_per_line]) for i in range(0, len(words), max_words_per_line)]
    return lines

def is_url(string):
    """Check if the given string is a valid HTTP/HTTPS URL."""
    try:
        result = urlparse(string)
        return result.scheme in ('http', 'https')
    except:
        return False

def download_captions(captions_url):
    """Download captions from the given URL."""
    try:
        logger.info(f"Downloading captions from URL: {captions_url}")
        response = requests.get(captions_url)
        response.raise_for_status()
        logger.info("Captions downloaded successfully.")
        return response.text
    except Exception as e:
        logger.error(f"Error downloading captions: {str(e)}")
        raise

def determine_alignment_code(position_str, alignment_str, x, y, video_width, video_height):
    """
    Determine the final \an alignment code and (x,y) position based on:
    - x,y (if provided)
    - position_str (one of top_left, top_center, ...)
    - alignment_str (left, center, right)
    - If x,y not provided, divide the video into a 3x3 grid and position accordingly.
    """
    logger.info(f"[determine_alignment_code] Inputs: position_str={position_str}, alignment_str={alignment_str}, x={x}, y={y}, video_width={video_width}, video_height={video_height}")

    horizontal_map = {
        'left': 1,
        'center': 2,
        'right': 3
    }

    # If x and y are provided, use them directly and set \an based on alignment_str
    if x is not None and y is not None:
        logger.info("[determine_alignment_code] x and y provided, ignoring position and alignment for grid.")
        vertical_code = 4  # Middle row
        horiz_code = horizontal_map.get(alignment_str, 2)  # Default to center
        an_code = vertical_code + (horiz_code - 1)
        logger.info(f"[determine_alignment_code] Using provided x,y. an_code={an_code}")
        return an_code, True, x, y

    # No x,y provided: determine position and alignment based on grid
    pos_lower = position_str.lower()
    if 'top' in pos_lower:
        vertical_base = 7  # Top row an codes start at 7
        vertical_center = video_height / 6
    elif 'middle' in pos_lower:
        vertical_base = 4  # Middle row an codes start at 4
        vertical_center = video_height / 2
    else:
        vertical_base = 1  # Bottom row an codes start at 1
        vertical_center = (5 * video_height) / 6

    if 'left' in pos_lower:
        left_boundary = 0
        right_boundary = video_width / 3
        center_line = video_width / 6
    elif 'right' in pos_lower:
        left_boundary = (2 * video_width) / 3
        right_boundary = video_width
        center_line = (5 * video_width) / 6
    else:
        # Center column
        left_boundary = video_width / 3
        right_boundary = (2 * video_width) / 3
        center_line = video_width / 2

    # Alignment affects horizontal position within the cell
    if alignment_str == 'left':
        final_x = left_boundary
        horiz_code = 1
    elif alignment_str == 'right':
        final_x = right_boundary
        horiz_code = 3
    else:
        final_x = center_line
        horiz_code = 2

    final_y = vertical_center
    an_code = vertical_base + (horiz_code - 1)

    logger.info(f"[determine_alignment_code] Computed final_x={final_x}, final_y={final_y}, an_code={an_code}")
    return an_code, True, int(final_x), int(final_y)

def create_style_line(style_options, video_resolution):
    """
    Create the style line for ASS subtitles.
    """
    font_family = style_options.get('font_family', 'Arial')
    available_fonts = get_available_fonts()

    # Case-insensitive font matching
    font_family_lower = font_family.lower()
    available_fonts_lower = [f.lower() for f in available_fonts]

    if font_family_lower not in available_fonts_lower:
        logger.warning(f"Font '{font_family}' not found in fontconfig.")
        return {'error': f"Font '{font_family}' not available.", 'available_fonts': available_fonts}

    # Use fontconfig-resolvable name in ASS (e.g. filename -> family when different)
    ass_font_name = resolve_font_family_for_ass(font_family)

    line_color = rgb_to_ass_color(style_options.get('line_color', '#FFFFFF'))
    secondary_color = line_color
    outline_color = rgb_to_ass_color(style_options.get('outline_color', '#000000'))
    # Support both back_color and box_color (back_color takes precedence)
    box_color = rgb_to_ass_color(style_options.get('back_color') or style_options.get('box_color', '#000000'))

    font_size = style_options.get('font_size', int(video_resolution[1] * 0.05))
    bold = '1' if style_options.get('bold', False) else '0'
    italic = '1' if style_options.get('italic', False) else '0'
    underline = '1' if style_options.get('underline', False) else '0'
    strikeout = '1' if style_options.get('strikeout', False) else '0'
    scale_x = style_options.get('scale_x', '100')
    scale_y = style_options.get('scale_y', '100')
    spacing = style_options.get('spacing', '0')
    angle = style_options.get('angle', '0')
    # ASS BorderStyle: 1 = outline+shadow (BackColour=shadow), 3 = opaque box (BackColour=box fill)
    use_box = style_options.get('box', False)
    border_style = '3' if use_box else str(style_options.get('border_style', 1))
    outline_width = style_options.get('outline_width', '2')
    shadow_offset = style_options.get('shadow_offset', '0')

    margin_l = style_options.get('margin_l', '20')
    margin_r = style_options.get('margin_r', '20')
    margin_v = style_options.get('margin_v', '20')

    # Default alignment in style (we override per event)
    alignment = 5

    style_line = (
        f"Style: Default,{ass_font_name},{font_size},{line_color},{secondary_color},"
        f"{outline_color},{box_color},{bold},{italic},{underline},{strikeout},"
        f"{scale_x},{scale_y},{spacing},{angle},{border_style},{outline_width},"
        f"{shadow_offset},{alignment},{margin_l},{margin_r},{margin_v},0"
    )
    logger.info(f"Created ASS style line: {style_line}")
    return style_line

def generate_ass_header(style_options, video_resolution):
    """
    Generate the ASS file header with the Default style.
    """
    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_resolution[0]}
PlayResY: {video_resolution[1]}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
"""
    style_line = create_style_line(style_options, video_resolution)
    if isinstance(style_line, dict) and 'error' in style_line:
        # Font-related error
        return style_line

    ass_header += style_line + "\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    logger.info("Generated ASS header.")
    return ass_header

### STYLE HANDLERS ###

def handle_classic(transcription_result, style_options, replace_dict, video_resolution):
    """
    Classic style handler: Centers the text based on position and alignment.
    """
    max_words_per_line = int(style_options.get('max_words_per_line', 0))
    all_caps = style_options.get('all_caps', False)
    if style_options['font_size'] is None:
        style_options['font_size'] = int(video_resolution[1] * 0.05)

    position_str = style_options.get('position', 'middle_center')
    alignment_str = style_options.get('alignment', 'center')
    x = style_options.get('x')
    y = style_options.get('y')

    an_code, use_pos, final_x, final_y = determine_alignment_code(
        position_str, alignment_str, x, y,
        video_width=video_resolution[0],
        video_height=video_resolution[1]
    )

    logger.info(f"[Classic] position={position_str}, alignment={alignment_str}, x={final_x}, y={final_y}, an_code={an_code}")

    events = []
    for segment in transcription_result['segments']:
        text = segment['text'].strip().replace('\n', ' ')
        lines = split_lines(text, max_words_per_line)
        processed_text = '\\N'.join(process_subtitle_text(line, replace_dict, all_caps, 0) for line in lines)
        start_time = format_ass_time(segment['start'])
        end_time = format_ass_time(segment['end'])
        position_tag = f"{{\\an{an_code}\\pos({final_x},{final_y})}}"
        events.append(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{position_tag}{processed_text}")
    logger.info(f"Handled {len(events)} dialogues in classic style.")
    return "\n".join(events)

def handle_karaoke(transcription_result, style_options, replace_dict, video_resolution):
    """
    Karaoke style handler: Highlights words as they are spoken.
    """
    max_words_per_line = int(style_options.get('max_words_per_line', 0))
    all_caps = style_options.get('all_caps', False)
    if style_options['font_size'] is None:
        style_options['font_size'] = int(video_resolution[1] * 0.05)

    position_str = style_options.get('position', 'middle_center')
    alignment_str = style_options.get('alignment', 'center')
    x = style_options.get('x')
    y = style_options.get('y')

    an_code, use_pos, final_x, final_y = determine_alignment_code(
        position_str, alignment_str, x, y,
        video_width=video_resolution[0],
        video_height=video_resolution[1]
    )
    word_color = rgb_to_ass_color(style_options.get('word_color', '#FFFF00'))

    logger.info(f"[Karaoke] position={position_str}, alignment={alignment_str}, x={final_x}, y={final_y}, an_code={an_code}")

    events = []
    for segment in transcription_result['segments']:
        words = segment.get('words', [])
        if not words:
            continue

        if max_words_per_line > 0:
            lines_content = []
            current_line = []
            current_line_words = 0
            for w_info in words:
                w = process_subtitle_text(w_info.get('word', ''), replace_dict, all_caps, 0)
                duration_cs = int(round((w_info['end'] - w_info['start']) * 100))
                highlighted_word = f"{{\\k{duration_cs}}}{w} "
                current_line.append(highlighted_word)
                current_line_words += 1
                if current_line_words >= max_words_per_line:
                    lines_content.append(''.join(current_line).strip())
                    current_line = []
                    current_line_words = 0
            if current_line:
                lines_content.append(''.join(current_line).strip())
        else:
            line_content = []
            for w_info in words:
                w = process_subtitle_text(w_info.get('word', ''), replace_dict, all_caps, 0)
                duration_cs = int(round((w_info['end'] - w_info['start']) * 100))
                highlighted_word = f"{{\\k{duration_cs}}}{w} "
                line_content.append(highlighted_word)
            lines_content = [''.join(line_content).strip()]

        dialogue_text = '\\N'.join(lines_content)
        start_time = format_ass_time(words[0]['start'])
        end_time = format_ass_time(words[-1]['end'])
        position_tag = f"{{\\an{an_code}\\pos({final_x},{final_y})}}"
        events.append(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{position_tag}{{\\c{word_color}}}{dialogue_text}")
    logger.info(f"Handled {len(events)} dialogues in karaoke style.")
    return "\n".join(events)

def handle_highlight(transcription_result, style_options, replace_dict, video_resolution):
    """
    Highlight style handler: Highlights words sequentially.
    """
    max_words_per_line = int(style_options.get('max_words_per_line', 0))
    all_caps = style_options.get('all_caps', False)
    if style_options['font_size'] is None:
        style_options['font_size'] = int(video_resolution[1] * 0.05)

    position_str = style_options.get('position', 'middle_center')
    alignment_str = style_options.get('alignment', 'center')
    x = style_options.get('x')
    y = style_options.get('y')

    an_code, use_pos, final_x, final_y = determine_alignment_code(
        position_str, alignment_str, x, y,
        video_width=video_resolution[0],
        video_height=video_resolution[1]
    )

    word_color = rgb_to_ass_color(style_options.get('word_color', '#FFFF00'))
    line_color = rgb_to_ass_color(style_options.get('line_color', '#FFFFFF'))
    events = []

    logger.info(f"[Highlight] position={position_str}, alignment={alignment_str}, x={final_x}, y={final_y}, an_code={an_code}")

    for segment in transcription_result['segments']:
        words = segment.get('words', [])
        if not words:
            continue

        # Process all words in the segment
        processed_words = []
        for w_info in words:
            w = process_subtitle_text(w_info.get('word', ''), replace_dict, all_caps, 0)
            if w:
                processed_words.append((w, w_info['start'], w_info['end']))

        if not processed_words:
            continue

        # Split into lines if max_words_per_line is specified
        if max_words_per_line > 0:
            line_sets = [processed_words[i:i+max_words_per_line] for i in range(0, len(processed_words), max_words_per_line)]
        else:
            line_sets = [processed_words]

        for line_set in line_sets:
            # Get the start time of the first word and end time of the last word
            line_start = line_set[0][1]
            line_end = line_set[-1][2]
            
            # Create a persistent line that stays visible during the entire segment
            base_text = ' '.join(word for word, _, _ in line_set)
            start_time = format_ass_time(line_start)
            end_time = format_ass_time(line_end)
            position_tag = f"{{\\an{an_code}\\pos({final_x},{final_y})}}"
            events.append(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{position_tag}{{\\c{line_color}}}{base_text}")
            
            # Add individual highlighting for each word
            for idx, (word, w_start, w_end) in enumerate(line_set):
                # Create the highlighted version of this word within the line
                highlighted_words = []
                
                for i, (w, _, _) in enumerate(line_set):
                    if i == idx:
                        # This is the current word - highlight it
                        highlighted_words.append(f"{{\\c{word_color}}}{w}{{\\c{line_color}}}")
                    else:
                        # Add the word without highlighting
                        highlighted_words.append(w)
                
                highlighted_text = ' '.join(highlighted_words)
                word_start_time = format_ass_time(w_start)
                word_end_time = format_ass_time(w_end)
                events.append(f"Dialogue: 1,{word_start_time},{word_end_time},Default,,0,0,0,,{position_tag}{{\\c{line_color}}}{highlighted_text}")

    logger.info(f"Handled {len(events)} dialogues in highlight style.")
    return "\n".join(events)

def handle_underline(transcription_result, style_options, replace_dict, video_resolution):
    """
    Underline style handler: Underlines the current word.
    """
    max_words_per_line = int(style_options.get('max_words_per_line', 0))
    all_caps = style_options.get('all_caps', False)
    if style_options['font_size'] is None:
        style_options['font_size'] = int(video_resolution[1] * 0.05)

    position_str = style_options.get('position', 'middle_center')
    alignment_str = style_options.get('alignment', 'center')
    x = style_options.get('x')
    y = style_options.get('y')

    an_code, use_pos, final_x, final_y = determine_alignment_code(
        position_str, alignment_str, x, y,
        video_width=video_resolution[0],
        video_height=video_resolution[1]
    )
    line_color = rgb_to_ass_color(style_options.get('line_color', '#FFFFFF'))
    events = []

    logger.info(f"[Underline] position={position_str}, alignment={alignment_str}, x={final_x}, y={final_y}, an_code={an_code}")

    for segment in transcription_result['segments']:
        words = segment.get('words', [])
        if not words:
            continue
        processed_words = []
        for w_info in words:
            w = process_subtitle_text(w_info.get('word', ''), replace_dict, all_caps, 0)
            if w:
                processed_words.append((w, w_info['start'], w_info['end']))

        if not processed_words:
            continue

        if max_words_per_line > 0:
            line_sets = [processed_words[i:i+max_words_per_line] for i in range(0, len(processed_words), max_words_per_line)]
        else:
            line_sets = [processed_words]

        for line_set in line_sets:
            for idx, (word, w_start, w_end) in enumerate(line_set):
                line_words = []
                for w_idx, (w_text, _, _) in enumerate(line_set):
                    if w_idx == idx:
                        line_words.append(f"{{\\u1}}{w_text}{{\\u0}}")
                    else:
                        line_words.append(w_text)
                full_text = ' '.join(line_words)
                start_time = format_ass_time(w_start)
                end_time = format_ass_time(w_end)
                position_tag = f"{{\\an{an_code}\\pos({final_x},{final_y})}}"
                events.append(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{position_tag}{{\\c{line_color}}}{full_text}")
    logger.info(f"Handled {len(events)} dialogues in underline style.")
    return "\n".join(events)

def handle_word_by_word(transcription_result, style_options, replace_dict, video_resolution):
    """
    Word-by-Word style handler: Displays each word individually.
    """
    max_words_per_line = int(style_options.get('max_words_per_line', 0))
    all_caps = style_options.get('all_caps', False)
    if style_options['font_size'] is None:
        style_options['font_size'] = int(video_resolution[1] * 0.05)

    position_str = style_options.get('position', 'middle_center')
    alignment_str = style_options.get('alignment', 'center')
    x = style_options.get('x')
    y = style_options.get('y')

    an_code, use_pos, final_x, final_y = determine_alignment_code(
        position_str, alignment_str, x, y,
        video_width=video_resolution[0],
        video_height=video_resolution[1]
    )
    word_color = rgb_to_ass_color(style_options.get('word_color', '#FFFF00'))
    events = []

    logger.info(f"[Word-by-Word] position={position_str}, alignment={alignment_str}, x={final_x}, y={final_y}, an_code={an_code}")

    for segment in transcription_result['segments']:
        words = segment.get('words', [])
        if not words:
            continue

        if max_words_per_line > 0:
            grouped_words = [words[i:i+max_words_per_line] for i in range(0, len(words), max_words_per_line)]
        else:
            grouped_words = [words]

        for word_group in grouped_words:
            for w_info in word_group:
                w = process_subtitle_text(w_info.get('word', ''), replace_dict, all_caps, 0)
                if not w:
                    continue
                start_time = format_ass_time(w_info['start'])
                end_time = format_ass_time(w_info['end'])
                position_tag = f"{{\\an{an_code}\\pos({final_x},{final_y})}}"
                events.append(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{position_tag}{{\\c{word_color}}}{w}")
    logger.info(f"Handled {len(events)} dialogues in word-by-word style.")
    return "\n".join(events)

STYLE_HANDLERS = {
    'classic': handle_classic,
    'karaoke': handle_karaoke,
    'highlight': handle_highlight,
    'underline': handle_underline,
    'word_by_word': handle_word_by_word
}

def srt_to_ass(transcription_result, style_type, settings, replace_dict, video_resolution):
    """
    Convert transcription result to ASS based on the specified style.
    """
    default_style_settings = {
        'line_color': '#FFFFFF',
        'word_color': '#FFFF00',
        'box_color': '#000000',
        'outline_color': '#000000',
        'all_caps': False,
        'max_words_per_line': 0,
        'font_size': None,
        'font_family': 'Arial',
        'bold': False,
        'italic': False,
        'underline': False,
        'strikeout': False,
        'outline_width': 2,
        'shadow_offset': 0,
        'border_style': 1,
        'box': False,
        'x': None,
        'y': None,
        'position': 'middle_center',
        'alignment': 'center'  # default alignment
    }
    style_options = {**default_style_settings, **settings}

    if style_options['font_size'] is None:
        style_options['font_size'] = int(video_resolution[1] * 0.05)

    ass_header = generate_ass_header(style_options, video_resolution)
    if isinstance(ass_header, dict) and 'error' in ass_header:
        # Font-related error
        return ass_header

    handler = STYLE_HANDLERS.get(style_type.lower())
    if not handler:
        logger.warning(f"Unknown style '{style_type}', defaulting to 'classic'.")
        handler = handle_classic

    dialogue_lines = handler(transcription_result, style_options, replace_dict, video_resolution)
    logger.info("Converted transcription result to ASS format.")
    return ass_header + dialogue_lines + "\n"

def process_subtitle_events(transcription_result, style_type, settings, replace_dict, video_resolution):
    """
    Process transcription results into ASS subtitle format.
    """
    return srt_to_ass(transcription_result, style_type, settings, replace_dict, video_resolution)

def parse_time_string(time_str):
    """Parse a time string in hh:mm:ss.ms or mm:ss.ms or ss.ms format to seconds (float)."""
    import re
    if not isinstance(time_str, str):
        raise ValueError("Time value must be a string in hh:mm:ss.ms format.")
    pattern = r"^(?:(\d+):)?(\d{1,2}):(\d{2}(?:\.\d{1,3})?)$"
    match = re.match(pattern, time_str)
    if not match:
        # Try ss.ms only
        try:
            return float(time_str)
        except Exception:
            raise ValueError(f"Invalid time string: {time_str}")
    h, m, s = match.groups(default="0")
    total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return total_seconds

def filter_subtitle_lines(sub_content, exclude_time_ranges, subtitle_type):
    """
    Remove subtitle lines/blocks that overlap with exclude_time_ranges.
    Supports 'ass' and 'srt' subtitle_type.
    """

    def parse_ass_time(ass_time):
        try:
            h, m, rest = ass_time.split(":")
            s, cs = rest.split(".")
            return int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100
        except Exception:
            return 0
    def parse_time_range(rng):
        start = parse_time_string(rng['start'])
        end = parse_time_string(rng['end'])
        return {'start': start, 'end': end}
    parsed_ranges = [parse_time_range(rng) for rng in exclude_time_ranges]
    if not exclude_time_ranges:
        return sub_content
    if subtitle_type == 'ass':
        lines = sub_content.splitlines()
        filtered_lines = []
        for line in lines:
            if line.startswith("Dialogue:"):
                parts = line.split(",", 10)
                if len(parts) > 3:
                    start = parse_ass_time(parts[1])
                    end = parse_ass_time(parts[2])
                    overlap = False
                    for rng in parsed_ranges:
                        if start < rng['end'] and end > rng['start']:
                            overlap = True
                            break
                    if overlap:
                        continue
            filtered_lines.append(line)
        return "\n".join(filtered_lines)
    elif subtitle_type == 'srt':
        subtitles = list(srt.parse(sub_content))
        filtered = []
        for sub in subtitles:
            start = sub.start.total_seconds()
            end = sub.end.total_seconds()
            overlap = False
            for rng in parsed_ranges:
                if start < rng['end'] and end > rng['start']:
                    overlap = True
                    break
            if not overlap:
                filtered.append(sub)
        return srt.compose(filtered)
    else:
        return sub_content

def normalize_exclude_time_ranges(exclude_time_ranges):
    norm = []
    for rng in exclude_time_ranges:
        start = rng.get("start")
        end = rng.get("end")
        if not isinstance(start, str) or not isinstance(end, str):
            raise ValueError("exclude_time_ranges start/end must be strings in hh:mm:ss.ms format.")
        start_sec = parse_time_string(start)
        end_sec = parse_time_string(end)
        if start_sec < 0 or end_sec < 0:
            raise ValueError("exclude_time_ranges start/end must be non-negative.")
        if end_sec <= start_sec:
            raise ValueError("exclude_time_ranges end must be strictly greater than start.")
        norm.append({"start": start, "end": end})
    return norm

def generate_ass_captions_v1(video_url, captions, settings, replace, exclude_time_ranges, job_id, language='auto', PlayResX=None, PlayResY=None):
    """
    Captioning process with transcription fallback and multiple styles.
    Integrates with the updated logic for positioning and alignment.
    If PlayResX and PlayResY are provided, use them for ASS generation; otherwise, get from video.
    """
    try:
        # Normalize exclude_time_ranges to ensure start/end are floats
        if exclude_time_ranges:
            exclude_time_ranges = normalize_exclude_time_ranges(exclude_time_ranges)

        if not isinstance(settings, dict):
            logger.error(f"Job {job_id}: 'settings' should be a dictionary.")
            return {"error": "'settings' should be a dictionary."}

        # Normalize keys by replacing hyphens with underscores
        style_options = {k.replace('-', '_'): v for k, v in settings.items()}

        if not isinstance(replace, list):
            logger.error(f"Job {job_id}: 'replace' should be a list of objects with 'find' and 'replace' keys.")
            return {"error": "'replace' should be a list of objects with 'find' and 'replace' keys."}

        # Convert 'replace' list to dictionary
        replace_dict = {}
        for item in replace:
            if 'find' in item and 'replace' in item:
                replace_dict[item['find']] = item['replace']
            else:
                logger.warning(f"Job {job_id}: Invalid replace item {item}. Skipping.")

        # Handle deprecated 'highlight_color' by merging it into 'word_color'
        if 'highlight_color' in style_options:
            logger.warning(f"Job {job_id}: 'highlight_color' is deprecated; merging into 'word_color'.")
            style_options['word_color'] = style_options.pop('highlight_color')

        # Check font availability
        font_family = style_options.get('font_family', 'Arial')
        available_fonts = get_available_fonts()

        # Case-insensitive font matching
        font_family_lower = font_family.lower()
        available_fonts_lower = [f.lower() for f in available_fonts]

        if font_family_lower not in available_fonts_lower:
            logger.warning(f"Job {job_id}: Font '{font_family}' not found in fontconfig.")
            # Return font error with available_fonts
            return {"error": f"Font '{font_family}' not available.", "available_fonts": available_fonts}

        logger.info(f"Job {job_id}: Font '{font_family}' is available.")

        # Determine if captions is a URL or raw content
        if captions and is_url(captions):
            logger.info(f"Job {job_id}: Captions provided as URL. Downloading captions.")
            try:
                captions_content = download_captions(captions)
            except Exception as e:
                logger.error(f"Job {job_id}: Failed to download captions: {str(e)}")
                return {"error": f"Failed to download captions: {str(e)}"}
        elif captions:
            logger.info(f"Job {job_id}: Captions provided as raw content.")
            captions_content = captions
        else:
            captions_content = None

        # Download the video
        try:
            video_path = download_file(video_url, LOCAL_STORAGE_PATH)
            logger.info(f"Job {job_id}: Video downloaded to {video_path}")
        except Exception as e:
            logger.error(f"Job {job_id}: Video download error: {str(e)}")
            # For non-font errors, do NOT include available_fonts
            return {"error": str(e)}

        # Get video resolution, unless provided
        if PlayResX is not None and PlayResY is not None:
            video_resolution = (PlayResX, PlayResY)
            logger.info(f"Job {job_id}: Using provided PlayResX/PlayResY = {PlayResX}x{PlayResY}")
        else:
            video_resolution = get_video_resolution(video_path)
            logger.info(f"Job {job_id}: Video resolution detected = {video_resolution[0]}x{video_resolution[1]}")

        # Determine style type
        style_type = style_options.get('style', 'classic').lower()
        logger.info(f"Job {job_id}: Using style '{style_type}' for captioning.")

        # Determine subtitle content
        if captions_content:
            # Check if it's ASS by looking for '[Script Info]'
            if '[Script Info]' in captions_content:
                # It's ASS directly
                subtitle_content = captions_content
                subtitle_type = 'ass'
                logger.info(f"Job {job_id}: Detected ASS formatted captions.")
            elif is_srt_format(captions_content):
                # It's SRT format
                logger.info(f"Job {job_id}: Detected SRT formatted captions.")
                # Validate style for SRT
                if style_type != 'classic':
                    error_message = "Only 'classic' style is supported for SRT captions."
                    logger.error(f"Job {job_id}: {error_message}")
                    return {"error": error_message}
                try:
                    transcription_result = srt_to_transcription_result(captions_content)
                except Exception as e:
                    logger.error(f"Job {job_id}: Error parsing SRT content: {str(e)}")
                    return {"error": f"Invalid SRT format: {str(e)}"}
                # Generate ASS based on chosen style
                subtitle_content = process_subtitle_events(transcription_result, style_type, style_options, replace_dict, video_resolution)
                subtitle_type = 'ass'
            else:
                # It's plain text - display for entire video duration
                logger.info(f"Job {job_id}: Detected plain text captions.")
                video_duration = get_video_duration(video_path)
                transcription_result = plain_text_to_transcription_result(captions_content, video_duration)
                # Generate ASS based on chosen style
                subtitle_content = process_subtitle_events(transcription_result, style_type, style_options, replace_dict, video_resolution)
                subtitle_type = 'ass'
        else:
            # No captions provided, generate transcription
            logger.info(f"Job {job_id}: No captions provided, generating transcription.")
            transcription_result = generate_transcription(video_path, language=language)
            # Generate ASS based on chosen style
            subtitle_content = process_subtitle_events(transcription_result, style_type, style_options, replace_dict, video_resolution)
            subtitle_type = 'ass'

        # Check for subtitle processing errors
        if isinstance(subtitle_content, dict) and 'error' in subtitle_content:
            logger.error(f"Job {job_id}: {subtitle_content['error']}")
            # Only include 'available_fonts' if it's a font-related error
            if 'available_fonts' in subtitle_content:
                return {"error": subtitle_content['error'], "available_fonts": subtitle_content.get('available_fonts', [])}
            else:
                return {"error": subtitle_content['error']}

        # After subtitle_content is generated and before saving to file:
        if exclude_time_ranges:
            subtitle_content = filter_subtitle_lines(subtitle_content, exclude_time_ranges, subtitle_type)
            if subtitle_type == 'ass':
                logger.info(f"Job {job_id}: Filtered ASS Dialogue lines due to exclude_time_ranges.")
            elif subtitle_type == 'srt':
                logger.info(f"Job {job_id}: Filtered SRT subtitle blocks due to exclude_time_ranges.")

        # Save the subtitle content
        subtitle_filename = f"{job_id}.{subtitle_type}"
        subtitle_path = os.path.join(LOCAL_STORAGE_PATH, subtitle_filename)
        try:
            with open(subtitle_path, 'w', encoding='utf-8') as f:
                f.write(subtitle_content)
            logger.info(f"Job {job_id}: Subtitle file saved to {subtitle_path}")
        except Exception as e:
            logger.error(f"Job {job_id}: Failed to save subtitle file: {str(e)}")
            return {"error": f"Failed to save subtitle file: {str(e)}"}

        return subtitle_path
    except Exception as e:
        logger.error(f"Job {job_id}: Error in generate_ass_captions_v1: {str(e)}", exc_info=True)
        return {"error": str(e)}
