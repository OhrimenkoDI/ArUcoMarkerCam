import json
from pathlib import Path
import re

import cv2
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SOURCE_DIR = Path("markers") / "aruco_single"
OUTPUT_PDF = SOURCE_DIR / "aruco_single_markers.pdf"
METADATA_PATH = SOURCE_DIR.parent / "marker_pack.json"

MARKER_PRINT_SIZE_MM = 80.0
PAGE_WIDTH_MM = 210.0
PAGE_HEIGHT_MM = 297.0
TOP_MARGIN_MM = 22.0
BOTTOM_MARGIN_MM = 30.0
CAPTION_GAP_MM = 12.0
TEXT_BLOCK_HEIGHT_MM = 18.0
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SIZE_IN_FILENAME_RE = re.compile(r"_(\d+(?:\.\d+)?)mm$", re.IGNORECASE)
ID_IN_FILENAME_RE = re.compile(r"_id(\d+)", re.IGNORECASE)


def mm_to_inches(value_mm: float) -> float:
    return value_mm / 25.4


def load_dictionary_name() -> str:
    if METADATA_PATH.exists():
        raw = json.loads(METADATA_PATH.read_text(encoding="utf-8")).get("dictionary_name", "")
        return raw.removeprefix("DICT_")
    return "ARUCO"


def collect_images(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def get_marker_size_mm(image_path: Path) -> float:
    match = SIZE_IN_FILENAME_RE.search(image_path.stem)
    if match:
        return float(match.group(1))
    return MARKER_PRINT_SIZE_MM


def draw_marker_page(pdf: PdfPages, image_path: Path, dict_name: str) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    marker_size_mm = get_marker_size_mm(image_path)

    page_width_in = mm_to_inches(PAGE_WIDTH_MM)
    page_height_in = mm_to_inches(PAGE_HEIGHT_MM)
    figure = plt.figure(figsize=(page_width_in, page_height_in))
    axes = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axes.set_xlim(0.0, PAGE_WIDTH_MM)
    axes.set_ylim(0.0, PAGE_HEIGHT_MM)
    axes.axis("off")

    available_height_mm = PAGE_HEIGHT_MM - TOP_MARGIN_MM - BOTTOM_MARGIN_MM - CAPTION_GAP_MM - TEXT_BLOCK_HEIGHT_MM
    marker_side_mm = min(marker_size_mm, PAGE_WIDTH_MM * 0.8, available_height_mm)
    x0 = (PAGE_WIDTH_MM - marker_side_mm) / 2.0
    y0 = BOTTOM_MARGIN_MM + TEXT_BLOCK_HEIGHT_MM + CAPTION_GAP_MM

    axes.imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=255,
        extent=(x0, x0 + marker_side_mm, y0, y0 + marker_side_mm),
        interpolation="nearest",
    )

    id_match = ID_IN_FILENAME_RE.search(image_path.stem)
    marker_id = int(id_match.group(1)) if id_match else "?"

    axes.text(
        PAGE_WIDTH_MM / 2.0,
        PAGE_HEIGHT_MM - 8.0,
        dict_name,
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        family="DejaVu Sans",
    )
    axes.text(
        PAGE_WIDTH_MM / 2.0,
        PAGE_HEIGHT_MM - 18.0,
        f"ID: {marker_id}",
        ha="center",
        va="center",
        fontsize=40,
        fontweight="bold",
        family="DejaVu Sans",
    )

    caption_y = BOTTOM_MARGIN_MM + 10.0
    axes.text(
        PAGE_WIDTH_MM / 2.0,
        caption_y + 7.0,
        image_path.name,
        ha="center",
        va="center",
        fontsize=12,
        family="DejaVu Sans",
    )
    axes.text(
        PAGE_WIDTH_MM / 2.0,
        caption_y,
        f"Marker size: {marker_size_mm:g} mm",
        ha="center",
        va="center",
        fontsize=11,
        family="DejaVu Sans",
    )

    pdf.savefig(figure, dpi=300)
    plt.close(figure)


def main() -> None:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR.resolve()}")

    image_paths = collect_images(SOURCE_DIR)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {SOURCE_DIR.resolve()}")

    dict_name = load_dictionary_name()
    with PdfPages(OUTPUT_PDF) as pdf:
        for image_path in image_paths:
            draw_marker_page(pdf, image_path, dict_name)

    print(f"Created PDF: {OUTPUT_PDF.resolve()}")
    print(f"Pages: {len(image_paths)}")
    print("Marker size: from filename suffix _XXXmm, fallback to default constant")


if __name__ == "__main__":
    main()
