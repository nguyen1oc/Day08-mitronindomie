"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft + Post-processing cấu trúc Header.
"""

import json
import re
import sys
from pathlib import Path

from markitdown import MarkItDown

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def clean_pdf_noise(text: str) -> str:
    """Loại bỏ các dòng rác header/footer của PDF."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Bỏ trang, ngày tháng header, URL footer
        if stripped == "\x0c" or re.match(r"^\d{2}/\d{2}/\d{4},\s+\d{2}:\d{2}", stripped):
            continue
        if re.search(r"https://www\.leagueoflegends\.com/.*?\d+/\d+", stripped):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def format_headers(text: str) -> str:
    """Tự động nhận diện và chuyển các dòng tiêu đề thành Markdown Headers (#, ##, ###, ####)."""
    lines = text.splitlines()
    result = []
    in_champions_section = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append(line)
            continue

        # Header 1: Tiêu đề lớn
        if stripped.startswith("THÔNG TIN CẬP NHẬT LMHT") or stripped.startswith("Patch "):
            result.append(f"# {stripped}")
            continue

        # Header 2: Section lớn (ALL CAPS hoặc các từ khóa chính)
        if stripped in ("TƯỚNG", "TIÊU ĐIỂM CẬP NHẬT", "CẬP NHẬT TRÒ CHƠI", "TRANG BỊ", "BẢNG CỘNG ĐỒNG") or (stripped.isupper() and len(stripped) < 60 and not stripped.startswith("http")):
            if stripped == "TƯỚNG":
                in_champions_section = True
            elif stripped in ("TIÊU ĐIỂM CẬP NHẬT", "TRANG BỊ", "CẬP NHẬT TRÒ CHƠI"):
                in_champions_section = False
            result.append(f"## {stripped}")
            continue

        # Header 4: Chiêu thức (Q -, W -, E -, R -, Nội Tại, Chỉ Số Cơ Bản)
        if re.match(r"^(Q|W|E|R|QQ|QW|QE|EQ|EW|EE)\s*-\s*", stripped) or stripped.startswith("Nội Tại") or stripped.startswith("Chỉ Số Cơ Bản"):
            result.append(f"#### {stripped}")
            continue

        # Header 3: Tên Tướng (Single/Double word khi nằm trong TƯỚNG section hoặc dạng tên tướng đứng độc lập)
        if in_champions_section and len(stripped.split()) <= 3 and not stripped.startswith("“") and not stripped.startswith("http") and ":" not in stripped and "⇒" not in stripped:
            result.append(f"### {stripped}")
            continue

        result.append(line)

    return "\n".join(result)


def enhance_markdown(text: str) -> str:
    """Làm sạch rác PDF và chuẩn hóa cấu trúc Header cho Markdown."""
    cleaned = clean_pdf_noise(text)
    enhanced = format_headers(cleaned)
    return enhanced


def convert_legal_docs() -> None:
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            result = md.convert(str(filepath))
            enhanced_text = enhance_markdown(result.text_content)
            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(enhanced_text, encoding="utf-8")
            print(f"  [OK] Saved (Enhanced with Headers): {output_path}")


def convert_news_articles() -> None:
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = output_dir / f"{filepath.stem}.md"

            header = f"# {data.get('title', 'Unknown')}\n\n"
            header += f"**Source:** {data.get('url', 'N/A')}\n"
            header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

            content = header + data.get("content_markdown", "")
            output_path.write_text(content, encoding="utf-8")
            print(f"  [OK] Saved: {output_path}")


def convert_all() -> None:
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown Enhanced)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n[OK] Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()

