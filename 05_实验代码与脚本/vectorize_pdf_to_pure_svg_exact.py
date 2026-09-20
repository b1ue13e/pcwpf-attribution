from __future__ import annotations

import argparse
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


def extract_rgb_from_pdf(
    pdf_path: Path, image_object: int = 6, smask_object: int = 7
) -> tuple[int, int, bytes]:
    pdf_bytes = pdf_path.read_bytes()

    img_obj = _extract_object(pdf_bytes, image_object)
    img_dict, img_stream = _extract_stream(img_obj)
    width = _dict_int(img_dict, "Width")
    height = _dict_int(img_dict, "Height")

    img_raw = zlib.decompress(img_stream)
    expected_rgb = width * height * 3
    if len(img_raw) != expected_rgb:
        raise ValueError(f"RGB payload size mismatch: {len(img_raw)} vs {expected_rgb}")

    # Keep soft-mask validation so dimensions are guaranteed to match source asset.
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
    if any(alpha != 255 for alpha in alpha_raw):
        raise ValueError(
            "Source contains non-opaque alpha. This script currently enforces fully opaque input."
        )

    return width, height, img_raw


def _iter_rgb_runs(
    rgb_bytes: bytes, width: int, height: int
) -> tuple[int, int, int, int, int, int]:
    row_stride = width * 3
    for y in range(height):
        row = rgb_bytes[y * row_stride : (y + 1) * row_stride]
        x = 0
        while x < width:
            i = x * 3
            r = row[i]
            g = row[i + 1]
            b = row[i + 2]
            x2 = x + 1
            j = i + 3
            while x2 < width and row[j] == r and row[j + 1] == g and row[j + 2] == b:
                x2 += 1
                j += 3
            yield y, x, x2 - x, r, g, b
            x = x2


def write_pure_vector_svg(
    rgb_bytes: bytes,
    width: int,
    height: int,
    output_svg: Path,
) -> int:
    run_count = 0
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    with output_svg.open("w", encoding="utf-8", newline="\n") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" shape-rendering="crispEdges">\n'
        )

        buffer: list[str] = []
        for y, x, run_len, r, g, b in _iter_rgb_runs(rgb_bytes, width, height):
            buffer.append(
                f'<rect x="{x}" y="{y}" width="{run_len}" height="1" fill="#{r:02x}{g:02x}{b:02x}"/>\n'
            )
            run_count += 1
            if len(buffer) >= 6000:
                f.write("".join(buffer))
                buffer.clear()

        if buffer:
            f.write("".join(buffer))
        f.write("</svg>\n")

    return run_count


def verify_svg_roundtrip(svg_path: Path, width: int, height: int, source_rgb_bytes: bytes) -> dict[str, str]:
    rect_pattern = re.compile(
        r'<rect x="(\d+)" y="(\d+)" width="(\d+)" height="1" fill="#([0-9a-fA-F]{6})"/>'
    )

    row_stride = width * 3
    current_y = 0
    current_x = 0
    row_buf = bytearray()
    reconstructed = hashlib.sha256()
    pixel_count = 0
    run_count = 0
    saw_any_rect = False

    with svg_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("<rect "):
                continue
            saw_any_rect = True
            match = rect_pattern.fullmatch(line.strip())
            if not match:
                raise ValueError(f"Unexpected rect format: {line[:120]!r}")

            x = int(match.group(1))
            y = int(match.group(2))
            run_len = int(match.group(3))
            rgb_hex = match.group(4)

            if y != current_y:
                if y != current_y + 1:
                    raise ValueError(f"Non-consecutive row index: got {y}, expected {current_y + 1}")
                if current_x != width:
                    raise ValueError(f"Row {current_y} width mismatch: {current_x} vs {width}")
                if len(row_buf) != row_stride:
                    raise ValueError(
                        f"Row {current_y} byte count mismatch: {len(row_buf)} vs {row_stride}"
                    )
                reconstructed.update(row_buf)
                row_buf.clear()
                current_y = y
                current_x = 0

            if y != current_y:
                raise ValueError(f"Row mismatch after transition: {y} vs {current_y}")
            if x != current_x:
                raise ValueError(f"X continuity mismatch at row {y}: got {x}, expected {current_x}")
            if run_len <= 0:
                raise ValueError(f"Invalid run length: {run_len}")
            if current_x + run_len > width:
                raise ValueError(f"Run overflows row at y={y}, x={x}, width={run_len}")

            rgb = bytes.fromhex(rgb_hex)
            row_buf.extend(rgb * run_len)
            current_x += run_len
            pixel_count += run_len
            run_count += 1

    if not saw_any_rect:
        raise ValueError("No vector rects found in generated SVG.")
    if current_x != width:
        raise ValueError(f"Last row width mismatch: {current_x} vs {width}")
    if len(row_buf) != row_stride:
        raise ValueError(f"Last row byte count mismatch: {len(row_buf)} vs {row_stride}")
    reconstructed.update(row_buf)
    if current_y != height - 1:
        raise ValueError(f"Last row index mismatch: {current_y} vs {height - 1}")

    source_sha = hashlib.sha256(source_rgb_bytes).hexdigest()
    reconstructed_sha = reconstructed.hexdigest()
    return {
        "source_rgb_sha256": source_sha,
        "reconstructed_rgb_sha256": reconstructed_sha,
        "hash_match": str(source_sha == reconstructed_sha),
        "pixel_count": str(pixel_count),
        "expected_pixel_count": str(width * height),
        "run_count": str(run_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a pure-vector, pixel-exact SVG from an image-only PDF and self-verify."
    )
    parser.add_argument(
        "--source",
        default=r"C:\Users\32110\Desktop\unwatermarked_Gemini_Generated_Image_1poah91poah91poa.pdf",
        help="Source PDF path.",
    )
    parser.add_argument(
        "--out-prefix",
        default="gemini_architecture_pure_vector_exact",
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

    width, height, rgb_bytes = extract_rgb_from_pdf(
        pdf_path=source,
        image_object=args.image_object,
        smask_object=args.smask_object,
    )

    svg_path = out_prefix.with_suffix(".svg")
    report_path = out_prefix.with_suffix(".txt")

    run_count = write_pure_vector_svg(rgb_bytes, width, height, svg_path)
    verify_info = verify_svg_roundtrip(svg_path, width, height, rgb_bytes)
    file_size = svg_path.stat().st_size

    report_lines = [
        f"source_pdf={source}",
        f"svg_output={svg_path}",
        f"size={width}x{height}",
        f"file_size_bytes={file_size}",
        f"run_count_written={run_count}",
        f"run_count_verified={verify_info['run_count']}",
        f"pixel_count_verified={verify_info['pixel_count']}",
        f"expected_pixel_count={verify_info['expected_pixel_count']}",
        f"source_rgb_sha256={verify_info['source_rgb_sha256']}",
        f"reconstructed_rgb_sha256={verify_info['reconstructed_rgb_sha256']}",
        f"hash_match={verify_info['hash_match']}",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8", newline="\n")

    print(f"Generated: {svg_path}")
    print(f"Generated: {report_path}")
    print(f"run_count={run_count}")
    print(f"hash_match={verify_info['hash_match']}")


if __name__ == "__main__":
    main()
