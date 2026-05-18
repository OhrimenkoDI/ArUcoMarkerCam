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
    # Загружаем изображение маркера в оттенках серого. Для ArUco это
    # предпочтительный вариант: сам маркер является черно-белой бинарной
    # матрицей, поэтому цветовые каналы не несут полезной информации.
    # Grayscale также помогает сохранить исходные значения пикселей без
    # лишних преобразований цвета.
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    # Определяем физический размер маркера для печати. В первую очередь размер
    # берется из имени файла, например из aruco_id03_180mm.png будет получено
    # значение 180 мм. Если в имени файла нет суффикса "_XXXmm", используется
    # значение по умолчанию из константы MARKER_PRINT_SIZE_MM.
    marker_size_mm = get_marker_size_mm(image_path)

    # Matplotlib задает размер figure в дюймах, а все константы разметки выше
    # описаны в миллиметрах, потому что это удобнее для печати. Поэтому сначала
    # переводим размер страницы A4 из миллиметров в дюймы. Если PDF печатать
    # без масштабирования, размеры на бумаге будут соответствовать расчетным.
    page_width_in = mm_to_inches(PAGE_WIDTH_MM)
    page_height_in = mm_to_inches(PAGE_HEIGHT_MM)
    figure = plt.figure(figsize=(page_width_in, page_height_in))

    # Создаем одну область рисования на всю страницу. Затем переводим ее
    # координатную систему в миллиметры: X идет от 0 до ширины A4, Y - от 0 до
    # высоты A4. Благодаря этому все координаты ниже задаются напрямую в
    # печатных единицах, без ручного пересчета в доли figure.
    axes = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axes.set_xlim(0.0, PAGE_WIDTH_MM)
    axes.set_ylim(0.0, PAGE_HEIGHT_MM)
    axes.axis("off")

    # Считаем доступную высоту под сам маркер. Из полной высоты страницы
    # вычитаются верхний отступ, нижний отступ, промежуток между маркером и
    # подписью, а также высота текстового блока снизу. Маркер должен поместиться
    # в оставшуюся область, иначе при печати он налезет на подписи или края.
    available_height_mm = PAGE_HEIGHT_MM - TOP_MARGIN_MM - BOTTOM_MARGIN_MM - CAPTION_GAP_MM - TEXT_BLOCK_HEIGHT_MM

    # Итоговая сторона маркера обычно равна запрошенному физическому размеру,
    # но дополнительно ограничивается шириной страницы и доступной высотой.
    # Ограничение PAGE_WIDTH_MM * 0.8 оставляет боковые поля, чтобы маркер не
    # печатался вплотную к краям листа даже при больших размерах из имени файла.
    marker_side_mm = min(marker_size_mm, PAGE_WIDTH_MM * 0.8, available_height_mm)

    # Вычисляем левый нижний угол прямоугольника, в который будет вписан
    # маркер. По X маркер центрируется на странице. По Y он поднимается над
    # нижним текстовым блоком: учитываются нижний отступ, высота подписи и
    # дополнительный зазор между подписью и изображением.
    x0 = (PAGE_WIDTH_MM - marker_side_mm) / 2.0
    y0 = BOTTOM_MARGIN_MM + TEXT_BLOCK_HEIGHT_MM + CAPTION_GAP_MM

    # Рисуем маркер в точный прямоугольник на PDF-странице. Параметр extent
    # задает реальные координаты этого прямоугольника в миллиметрах. Интерполяция
    # "nearest" отключает сглаживание границ между черными и белыми клетками:
    # это важно, потому что размытые края могут ухудшить распознавание ArUco
    # после печати и последующей съемки камерой.
    axes.imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=255,
        extent=(x0, x0 + marker_side_mm, y0, y0 + marker_side_mm),
        interpolation="nearest",
    )

    # Достаем числовой ID маркера из имени файла, например из
    # aruco_id21_285mm.png будет получено значение 21. Если файл назван иначе и
    # ID не найден, используем "?", чтобы генерация PDF не прерывалась из-за
    # нестандартного имени.
    id_match = ID_IN_FILENAME_RE.search(image_path.stem)
    marker_id = int(id_match.group(1)) if id_match else "?"

    # Верхние подписи привязываются не к краю страницы, а к фактическому верху
    # маркера. Так они всегда остаются максимально близко к коду даже тогда,
    # когда размер маркера меняется от файла к файлу.
    marker_top_mm = y0 + marker_side_mm
    id_y = marker_top_mm + 9.0
    dict_y = id_y + 14.0

    # Сначала печатаем имя словаря ArUco, ниже - крупный ID маркера. Большой ID
    # нужен для быстрой визуальной сортировки листов: не нужно каждый раз
    # сканировать сам маркер, чтобы понять его номер.
    axes.text(
        PAGE_WIDTH_MM / 2.0,
        dict_y,
        dict_name,
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        family="DejaVu Sans",
    )
    axes.text(
        PAGE_WIDTH_MM / 2.0,
        id_y,
        f"ID: {marker_id}",
        ha="center",
        va="center",
        fontsize=40,
        fontweight="bold",
        family="DejaVu Sans",
    )

    # Нижний блок подписи содержит имя исходного файла и расчетный физический
    # размер маркера. Эта информация полезна после печати, нарезки или
    # перемешивания листов: по подписи можно восстановить, из какого файла был
    # напечатан маркер и какой размер должен быть у его стороны.
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

    # Добавляем готовую A4-страницу в уже открытый PDF-файл. После сохранения
    # закрываем figure, чтобы Matplotlib освободил память. Это особенно важно,
    # когда в папке много маркеров и функция вызывается по одному разу на
    # каждое изображение.
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
