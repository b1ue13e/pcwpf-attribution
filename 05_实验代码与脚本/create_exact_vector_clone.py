from __future__ import annotations

import argparse
import base64
import hashlib
import re
import zlib
from pathlib import Path

from PIL import Image


def _extract_object(pdf_bytes: bytes, object_number: int) -> bytes:
    pattern = re.compile(rb"%d\s+0\s+obj\s*(.*?)endobj" % object_number, re.S)
    match = pattern.search(pdf_bytes)
    if not match:
        raise ValueError(f"Could not find object {object_number} in PDF.")
    return match.group(1)


def _extract_stream(obj_bytes: bytes) -> tuple[bytes, bytes]:
    stream_index = obj_bytes.find(b"stream")
    if stream_index < 0:
        raise ValueError("Object has no stream section.")

    dict_part = obj_bytes[:stream_index].strip()
    stream_part = obj_bytes[stream_index + len(b"stream") :]

    if stream_part.startswith(b"\r\n"):
        stream_part = stream_part[2:]
    elif stream_part.startswith(b"\n"):
        stream_part = stream_part[1:]

    end_index = stream_part.find(b"endstream")
    if end_index < 0:
        raise ValueError("Stream does not contain endstream marker.")

    stream_data = stream_part[:end_index]
    if stream_data.endswith(b"\r\n"):
        stream_data = stream_data[:-2]
    elif stream_data.endswith(b"\n"):
        stream_data = stream_data[:-1]

    return dict_part, stream_data


def _dict_int(dict_bytes: bytes, key: str) -> int:
    match = re.search(rb"/%s\s+(\d+)" % key.encode("ascii"), dict_bytes)
    if not match:
        raise ValueError(f"Missing /{key} in object dictionary.")
    return int(match.group(1))


def extract_rgba_from_pdf(pdf_path: Path, image_object: int = 6, smask_object: int = 7) -> Image.Image:
    pdf_bytes = pdf_path.read_bytes()

    img_obj = _extract_object(pdf_bytes, image_object)
    img_dict, img_stream = _extract_stream(img_obj)
    width = _dict_int(img_dict, "Width")
    height = _dict_int(img_dict, "Height")

    img_raw = zlib.decompress(img_stream)
    expected_rgb = width * height * 3
    if len(img_raw) != expected_rgb:
        raise ValueError(f"RGB payload size mismatch: {len(img_raw)} vs {expected_rgb}")
    rgb = Image.frombytes("RGB", (width, height), img_raw)

    mask_obj = _extract_object(pdf_bytes, smask_object)
    mask_dict, mask_stream = _extract_stream(mask_obj)
    mask_width = _dict_int(mask_dict, "Width")
    mask_height = _dict_int(mask_dict, "Height")

    if (mask_width, mask_height) != (width, height):
        raise ValueError(
            f"Mask size mismatch: {(mask_width, mask_height)} vs {(width, height)}"
        )

    alpha_raw = zlib.decompress(mask_stream)
    expected_alpha = width * height
    if len(alpha_raw) != expected_alpha:
        raise ValueError(f"Alpha payload size mismatch: {len(alpha_raw)} vs {expected_alpha}")

    alpha = Image.frombytes("L", (width, height), alpha_raw)
    rgba = rgb.convert("RGBA")
    rgba.putalpha(alpha)
    return rgba


def write_embedded_svg(image_png_bytes: bytes, width: int, height: int, output_svg: Path) -> None:
    b64 = base64.b64encode(image_png_bytes).decode("ascii")
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <image x="0" y="0" width="{width}" height="{height}"
         href="data:image/png;base64,{b64}" />
</svg>
"""
    output_svg.write_text(svg, encoding="utf-8", newline="\n")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a pixel-identical SVG clone from an image-only PDF."
    )
    parser.add_argument(
        "--source",
        default=r"C:\Users\32110\Desktop\unwatermarked_Gemini_Generated_Image_1poah91poah91poa.pdf",
        help="Source PDF path.",
    )
    parser.add_argument(
        "--out-prefix",
        default="gemini_architecture_exact_clone",
        help="Output file prefix (without extension).",
    )
    parser.add_argument(
        "--image-object",
        type=int,
        default=6,
        help="PDF object number of RGB image stream.",
    )
    parser.add_argument(
        "--smask-object",
        type=int,
        default=7,
        help="PDF object number of alpha mask stream.",
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source PDF does not exist: {source}")

    out_prefix = Path(args.out_prefix)
    if not out_prefix.is_absolute():
        out_prefix = Path.cwd() / out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    rgba = extract_rgba_from_pdf(
        pdf_path=source,
        image_object=args.image_object,
        smask_object=args.smask_object,
    )

    png_path = out_prefix.with_suffix(".png")
    svg_path = out_prefix.with_suffix(".svg")
    report_path = out_prefix.with_suffix(".txt")

    rgba.save(png_path)
    png_bytes = png_path.read_bytes()
    write_embedded_svg(png_bytes, rgba.width, rgba.height, svg_path)

    # Validation: extract base64 payload from generated SVG and hash-compare.
    svg_text = svg_path.read_text(encoding="utf-8")
    match = re.search(r"data:image/png;base64,([A-Za-z0-9+/=]+)", svg_text)
    if not match:
        raise ValueError("Could not find embedded PNG payload in generated SVG.")
    embedded_png_bytes = base64.b64decode(match.group(1))

    raw_rgba = rgba.tobytes()
    report_lines = [
        f"source_pdf={source}",
        f"png_output={png_path}",
        f"svg_output={svg_path}",
        f"size={rgba.width}x{rgba.height}",
        f"png_sha256={sha256_hex(png_bytes)}",
        f"embedded_png_sha256={sha256_hex(embedded_png_bytes)}",
        f"raw_rgba_sha256={sha256_hex(raw_rgba)}",
        f"embedded_payload_matches_png={embedded_png_bytes == png_bytes}",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8", newline="\n")

    print(f"Generated: {png_path}")
    print(f"Generated: {svg_path}")
    print(f"Generated: {report_path}")
    print(f"Embedded payload matches PNG: {embedded_png_bytes == png_bytes}")


if __name__ == "__main__":
    main()
