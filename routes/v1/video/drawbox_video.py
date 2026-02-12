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


from flask import Blueprint
from app_utils import validate_payload, queue_task_wrapper
import logging
import os
from services.v1.video.drawbox import drawbox_video
from services.authentication import authenticate
from services.cloud_storage import upload_file

v1_video_drawbox_bp = Blueprint('v1_video/drawbox', __name__)
logger = logging.getLogger(__name__)


@v1_video_drawbox_bp.route('/v1/video/drawbox', methods=['POST'])
@authenticate
@validate_payload({
    "type": "object",
    "properties": {
        "video_url": {"type": "string", "format": "uri"},
        "box_x": {"oneOf": [{"type": "integer"}, {"type": "string"}]},
        "box_y": {"oneOf": [{"type": "integer"}, {"type": "string"}]},
        "box_w": {"type": "string"},
        "box_h": {"type": "string"},
        "box_color": {"type": "string"},
        "box_overlay": {"type": "string"},
        "box_t": {"type": "string"},
        "text_x": {"oneOf": [{"type": "integer"}, {"type": "string"}]},
        "text_y": {"oneOf": [{"type": "integer"}, {"type": "string"}]},
        "text": {"type": "string"},
        "font": {"type": "string"},
        "text_size": {"type": "integer"},
        "fontsize": {"type": "integer"},
        "fontcolor": {"type": "string"},
        "text_align": {"type": "string", "enum": ["left", "center", "right"]},
        "text_max_width_margin": {"type": "integer"},
        "text_truncate_enabled": {"type": "boolean"},
        "webhook_url": {"type": "string", "format": "uri"},
        "id": {"type": "string"}
    },
    "required": ["video_url", "box_x", "box_y", "box_w", "box_h", "text_x", "text_y", "text", "font"],
    "additionalProperties": False
})
@queue_task_wrapper(bypass_queue=False)
def video_drawbox(job_id, data):
    """
    Draw a box and text on the video using FFmpeg drawbox and drawtext filters.
    Returns the URL of the processed video in cloud storage.
    """
    video_url = data['video_url']
    box_x = data.get('box_x', 0)
    box_y = data.get('box_y', 0)
    box_w = data.get('box_w', 'iw')
    box_h = data.get('box_h', 'ih')
    box_color = data.get('box_color', 'black')
    box_overlay = data.get('box_overlay', '1.0')
    box_t = data.get('box_t', 'fill')
    text_x = data.get('text_x', 0)
    text_y = data.get('text_y', 0)
    text = data.get('text', '')
    font = data.get('font', '')
    fontsize = data.get('text_size') or data.get('fontsize', 24)
    fontcolor = data.get('fontcolor', 'white')
    text_align = data.get('text_align', 'center')
    text_max_width_margin = data.get('text_max_width_margin')
    text_truncate_enabled = data.get('text_truncate_enabled', False)

    logger.info(f"Job {job_id}: Drawbox request for {video_url}")

    try:
        output_path = drawbox_video(
            video_url=video_url,
            box_x=box_x,
            box_y=box_y,
            box_w=box_w,
            box_h=box_h,
            text_x=text_x,
            text_y=text_y,
            text=text,
            font=font,
            job_id=job_id,
            box_color=box_color,
            box_overlay=box_overlay,
            box_t=box_t,
            fontsize=fontsize,
            fontcolor=fontcolor,
            text_align=text_align,
            text_max_width_margin=text_max_width_margin,
            text_truncate_enabled=text_truncate_enabled,
        )
        cloud_url = upload_file(output_path)
        logger.info(f"Job {job_id}: Uploaded to {cloud_url}")

        if os.path.exists(output_path):
            os.remove(output_path)
        out_dir = os.path.dirname(output_path)
        if os.path.isdir(out_dir) and not os.listdir(out_dir):
            try:
                os.rmdir(out_dir)
            except OSError:
                pass

        return cloud_url, "/v1/video/drawbox", 200
    except ValueError as e:
        logger.warning(f"Job {job_id}: {e}")
        return {"error": str(e)}, "/v1/video/drawbox", 400
    except Exception as e:
        logger.error(f"Job {job_id}: Drawbox failed - {str(e)}", exc_info=True)
        return {"error": str(e)}, "/v1/video/drawbox", 500
