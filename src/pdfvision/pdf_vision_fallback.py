import argparse
import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import fitz

VisionCallable = Callable[[List[Dict]], str]


@dataclass
class PdfExtractResult:
    text: str
    page_texts: List[str]
    fallback_page_images_b64: List[str]


def _b64encode_png(pix: fitz.Pixmap) -> str:
    return base64.b64encode(pix.tobytes("png")).decode("ascii")


def _union_rects(rects: Sequence[fitz.Rect]) -> Optional[fitz.Rect]:
    if not rects:
        return None
    u = fitz.Rect(rects[0])
    for r in rects[1:]:
        u |= r
    return u


def _content_bbox(page: fitz.Page, pad: float = 6.0) -> fitz.Rect:
    rects: List[fitz.Rect] = []
    try:
        for d in page.get_drawings():
            r = d.get("rect")
            if r and r.width > 0 and r.height > 0:
                rects.append(r)
    except Exception:
        pass

    box = _union_rects(rects) or page.rect

    return fitz.Rect(
        max(page.rect.x0, box.x0 - pad),
        max(page.rect.y0, box.y0 - pad),
        min(page.rect.x1, box.x1 + pad),
        min(page.rect.y1, box.y1 + pad),
    )


def _render_page_png_b64(
    page: fitz.Page,
    *,
    dpi: int = 300,
    clip_to_content: bool = True,
    pad: float = 6.0,
) -> str:
    clip_rect = _content_bbox(page, pad=pad) if clip_to_content else None
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, clip=clip_rect, alpha=False)
    return _b64encode_png(pix)


def _to_data_url_png(b64_png: str) -> str:
    return f"data:image/png;base64,{b64_png}"



def extract_pdf_text_and_fallback_images(
    pdf_bytes: bytes,
    *,
    dpi: int = 300,
    clip_to_content: bool = True,
    include_page_texts: bool = True,
) -> PdfExtractResult:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    page_texts: List[str] = []
    fallback_imgs: List[str] = []
    all_text_parts: List[str] = []

    for pno in range(doc.page_count):
        page = doc.load_page(pno)

        t = page.get_text("text") or ""
        t_stripped = t.strip()

        if include_page_texts:
            page_texts.append(t)

        if t_stripped:
            all_text_parts.append(t)
        else:
            b64_png = _render_page_png_b64(page, dpi=dpi, clip_to_content=clip_to_content)
            fallback_imgs.append(b64_png)

    full_text = "\n\n".join([p.strip() for p in all_text_parts if p and p.strip()]).strip()

    return PdfExtractResult(
        text=full_text,
        page_texts=page_texts if include_page_texts else [],
        fallback_page_images_b64=fallback_imgs,
    )


def extract_pdf_from_b64_and_fallback_images(
    pdf_b64: str,
    *,
    dpi: int = 300,
    clip_to_content: bool = True,
) -> PdfExtractResult:
    pdf_bytes = base64.b64decode(pdf_b64)
    return extract_pdf_text_and_fallback_images(
        pdf_bytes,
        dpi=dpi,
        clip_to_content=clip_to_content,
        include_page_texts=True,
    )



def vision_transcribe_pages(
    page_images_b64: List[str],
    *,
    vision_call: VisionCallable,
    prompt_text: str = "Transcribe all readable text from this invoice page. Output plain text only.",
    max_pages: Optional[int] = None,
) -> str:
    out: List[str] = []
    imgs = page_images_b64 if max_pages is None else page_images_b64[:max_pages]

    for b64_png in imgs:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": _to_data_url_png(b64_png)}},
                ],
            }
        ]
        out.append(vision_call(messages) or "")

    return "\n\n".join([t.strip() for t in out if t and t.strip()]).strip()


def extract_pdf_text_with_vision_fallback(
    pdf_b64: str,
    *,
    vision_call: VisionCallable,
    dpi: int = 300,
    clip_to_content: bool = True,
    prompt_text: str = "Transcribe all readable text from this invoice page. Output plain text only.",
    max_pages: Optional[int] = None,
) -> Tuple[str, PdfExtractResult]:
    res = extract_pdf_from_b64_and_fallback_images(pdf_b64, dpi=dpi, clip_to_content=clip_to_content)

    if not res.fallback_page_images_b64:
        return res.text, res

    ocr_text = vision_transcribe_pages(
        res.fallback_page_images_b64,
        vision_call=vision_call,
        prompt_text=prompt_text,
        max_pages=max_pages,
    )

    if res.text and ocr_text:
        final = (res.text + "\n\n" + ocr_text).strip()
    else:
        final = (res.text or ocr_text or "").strip()

    return final, res



def _write_png_b64_to_file(b64_png: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(b64_png))


def _load_pdf_bytes(pdf_path: Path) -> bytes:
    return pdf_path.read_bytes()


def _default_out_dir(pdf_path: Path) -> Path:
    return pdf_path.parent / f"{pdf_path.stem}__vision_fallback_out"


def _summarize_result(res: PdfExtractResult, pdf_path: Path) -> Dict:
    total_pages = fitz.open(pdf_path).page_count
    extracted_pages = sum(1 for t in res.page_texts if (t or "").strip())
    blank_pages = total_pages - extracted_pages
    return {
        "pdf": str(pdf_path),
        "total_pages": total_pages,
        "pages_with_text": extracted_pages,
        "pages_blank_text": blank_pages,
        "fallback_images_count": len(res.fallback_page_images_b64),
        "extracted_text_chars": len(res.text or ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PDF text extraction with full-page image fallback for scanned/image-only pages."
    )
    parser.add_argument("pdf", type=str, help="Path to a PDF file to test.")
    parser.add_argument(
        "--mode",
        choices=["render", "ocr"],
        default="render",
        help="render = render fallback page images; ocr = also run OCR using a vision adapter (optional).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for rendered page images.")
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Do not clip to content bbox; render full page area.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory for rendered images and summary; default is рядом with PDF.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Limit number of fallback pages to OCR (0 = no limit).",
    )
    parser.add_argument(
        "--write-text",
        action="store_true",
        help="Write extracted/ocr text to out-dir/text.txt",
    )
    parser.add_argument(
        "--summary-json",
        action="store_true",
        help="Also write summary.json to out-dir",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        return 2

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else _default_out_dir(pdf_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_bytes = _load_pdf_bytes(pdf_path)

    res = extract_pdf_text_and_fallback_images(
        pdf_bytes,
        dpi=args.dpi,
        clip_to_content=(not args.no_clip),
        include_page_texts=True,
    )

    summary = _summarize_result(res, pdf_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for i, b64_png in enumerate(res.fallback_page_images_b64, start=1):
        img_path = out_dir / f"fallback_page_{i:03d}.png"
        _write_png_b64_to_file(b64_png, img_path)

    if args.summary_json:
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    final_text = res.text
    if args.mode == "ocr":
        adapter = os.getenv("PDF_VISION_ADAPTER", "").strip().lower()

        if not adapter:
            print(
                "OCR mode requested, but no adapter set.\n"
                "Set environment variable PDF_VISION_ADAPTER=llama32 or PDF_VISION_ADAPTER=pixtral\n"
                "and ensure that callable can be imported in this environment.\n"
                "For now, OCR is skipped."
            )
        else:
            vision_call: Optional[VisionCallable] = None

            if adapter == "llama32":
                from llama32 import llama32 as vision_call  # type: ignore
            else:
                print(f"Unknown PDF_VISION_ADAPTER='{adapter}'. Supported: llama32, pixtral")
                vision_call = None

            if vision_call and res.fallback_page_images_b64:
                max_pages = args.max_pages if args.max_pages > 0 else None
                ocr_text = vision_transcribe_pages(
                    res.fallback_page_images_b64,
                    vision_call=vision_call,
                    prompt_text="Transcribe all readable text from this invoice page. Output plain text only.",
                    max_pages=max_pages,
                )
                if final_text and ocr_text:
                    final_text = (final_text + "\n\n" + ocr_text).strip()
                else:
                    final_text = (final_text or ocr_text or "").strip()

    if args.write_text:
        (out_dir / "text.txt").write_text(final_text or "", encoding="utf-8")

    print(f"\nWrote outputs to: {out_dir}")
    if res.fallback_page_images_b64:
        print(f"Rendered {len(res.fallback_page_images_b64)} fallback page image(s).")
    else:
        print("No fallback pages detected (PDF appears to have extractable text).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
