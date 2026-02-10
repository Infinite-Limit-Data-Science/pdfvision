import argparse
import base64
import importlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any

import fitz

VisionCallable = Callable[[List[Dict]], str]

INVOICE_EXTRACT_SYSTEM_PROMPT = """\
You are InvoiceJSONExtractor, a strict information extraction system.

You will be shown one or more images (pages) from a PDF attachment.
Your job is to extract invoice-related data and output EXACTLY ONE JSON object.

CRITICAL OUTPUT RULES:
- If the document is NOT an invoice, output EXACTLY:
  The image is not an invoice
  (and nothing else)
- Otherwise, output EXACTLY one JSON object surrounded by a fenced code block:
  ```json
  { ... }
  ```
- Do NOT output any commentary, explanation, markdown (other than the required ```json fence), or multiple JSON objects.
- Do NOT add extra top-level keys beyond the required schema below.
- All values MUST be strings. If missing/unknown, use "" (empty string). Never use null.

REQUIRED JSON SCHEMA (top-level keys):
- "invoice_date"
- "invoice_number"
- "gross_invoice_amount"
- "invoice_tax"
- "invoice_freight"
- "po_number"
- "po_line_number"      (outdated field: ALWAYS "")
- "po_line_amount"      (outdated field: ALWAYS "")
- "invoice_description"
- "invoice_items"       (array of objects; see below)

INVOICE_ITEMS SCHEMA (each object):
- "item_number"         (ALWAYS "" per requirements)
- "item_description"
- "item_quantity"
- "item_unit_price"
- "item_total"

Filtering:
- Only include line items where item_total is non-zero (item_total != "0" and not "0.00").
- Do not treat subtotal/tax/freight/total/amount-due lines as line items.

FIELD EXTRACTION RULES:
1) invoice_date:
   - Prefer the date in the header labeled "Invoice Date", "Date", or similar.
   - If multiple dates exist, DO NOT pick "Due Date" / "Payment Due Date".
   - If you cannot confidently identify invoice date, set "".

2) invoice_number:
   - ONLY extract if explicitly labeled "Invoice Number", "Invoice #", "Invoice No.", etc.
   - Return ONLY the identifier itself (no label text).
   - Never use PO number, account number, statement number, patient record number, etc.

3) gross_invoice_amount:
   - The grand total / invoice total / amount due for the invoice (the overall total).
   - Remove currency symbols and commas (e.g., "$1,234.56" -> "1234.56").
   - If multiple totals exist, prefer "Total" / "Invoice Total" / "Amount Due" over subtotals.

4) invoice_tax:
   - Extract the tax AMOUNT (not a percent). If only percent is shown, set "".

5) invoice_freight:
   - Freight/shipping/handling amount (not from a line item). If not shown, "".

6) po_number:
   - Only if clearly labeled with PO terms:
     "Purchase Order", "PO Number", "Customer PO", "PO #", etc.
   - Never use job numbers.

7) invoice_description:
   - Short summary of goods/services. If not clear, "".

FORMATTING RULES:
- Strip control characters.
- Numbers should be digits with optional decimal point (e.g., "100.00").
- Dates: keep the format as shown on the invoice (do not invent).

FEW-SHOT EXAMPLE 1 (tricky dates + multiple numbers):
Input (conceptual):
  Header shows:
    INVOICE #: INV-2026-00123
    Invoice Date: 01/15/2026
    Due Date: 02/14/2026
    PO #: 8100123456
    Total Amount Due: $1,234.56
  Items:
    1  Consulting Services    Qty 1   Unit $1000.00   Total $1000.00
    2  Support Plan           Qty 1   Unit $234.56    Total $234.56
Output:
```json
{
  "invoice_date": "01/15/2026",
  "invoice_number": "INV-2026-00123",
  "gross_invoice_amount": "1234.56",
  "invoice_tax": "",
  "invoice_freight": "",
  "po_number": "8100123456",
  "po_line_number": "",
  "po_line_amount": "",
  "invoice_description": "Consulting Services; Support Plan",
  "invoice_items": [
    {"item_number": "", "item_description": "Consulting Services", "item_quantity": "1", "item_unit_price": "1000.00", "item_total": "1000.00"},
    {"item_number": "", "item_description": "Support Plan", "item_quantity": "1", "item_unit_price": "234.56", "item_total": "234.56"}
  ]
}
```

FEW-SHOT EXAMPLE 2 (invoice number not explicitly labeled; must leave invoice_number blank):
Input (conceptual):
  Header shows:
    "Statement No: 555-ABC"   (NOT labeled as Invoice Number)
    Date: 03/01/2026
    Total: $50.00
  No explicit "Invoice #" label anywhere.
Output (invoice_number must be blank):
```json
{
  "invoice_date": "03/01/2026",
  "invoice_number": "",
  "gross_invoice_amount": "50.00",
  "invoice_tax": "",
  "invoice_freight": "",
  "po_number": "",
  "po_line_number": "",
  "po_line_amount": "",
  "invoice_description": "",
  "invoice_items": []
}
```

FEW-SHOT EXAMPLE 3 (filter out $0 line items; do not include zero totals):
Input (conceptual):
  Header shows:
    Invoice Number: INV-00077
    Invoice Date: 04/10/2026
    PO #: 8100999999
    Total: $120.00

  Line items:
    A) Subscription Credit
       Qty: 1
       Unit: $0.00
       Total: $0.00    (IGNORE)

    B) Monthly Service
       Qty: 1
       Unit: $120.00
       Total: $120.00  (KEEP)

Output (only include non-zero total items):
```json
{
  "invoice_date": "04/10/2026",
  "invoice_number": "INV-00077",
  "gross_invoice_amount": "120.00",
  "invoice_tax": "",
  "invoice_freight": "",
  "po_number": "8100999999",
  "po_line_number": "",
  "po_line_amount": "",
  "invoice_description": "Monthly Service",
  "invoice_items": [
    {
      "item_number": "",
      "item_description": "Monthly Service",
      "item_quantity": "1",
      "item_unit_price": "120.00",
      "item_total": "120.00"
    }
  ]
}
```
"""

INVOICE_VERIFY_SYSTEM_PROMPT = """\
You are InvoiceJSONVerifier.

You will be shown the same invoice pages plus a candidate JSON extraction.

Your task:
- Verify each non-empty field is supported by visible text on the pages.
- If a field is not clearly supported, set it to "".
- For any remaining non-empty field, provide short evidence text and page number.

CRITICAL:
- Output EXACTLY one JSON object surrounded by ```json fences.
- Keep the SAME required schema as the candidate extraction.
- You MAY add one extra key: "_evidence" (object).
- Do NOT add any other keys.

"_evidence" format:
{
  "invoice_number": {"page": "1", "evidence": "Invoice # INV-2026-00123"},
  ...
}
- Evidence strings must be short (<= 140 chars).
- Page numbers are 1-based and must be strings.

If the document is NOT an invoice, output EXACTLY:
The image is not an invoice
"""

INVOICE_KEYS = [
    "invoice_date",
    "invoice_number",
    "gross_invoice_amount",
    "invoice_tax",
    "invoice_freight",
    "po_number",
    "po_line_number",
    "po_line_amount",
    "invoice_description",
    "invoice_items",
]

ITEM_KEYS = [
    "item_number",
    "item_description",
    "item_quantity",
    "item_unit_price",
    "item_total",
]

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


def render_all_pdf_pages_as_images_b64(
    pdf_bytes: bytes,
    *,
    dpi: int = 300,
    clip_to_content: bool = True,
    max_pages: Optional[int] = None,
) -> List[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out: List[str] = []
    for pno in range(doc.page_count):
        if max_pages is not None and len(out) >= max_pages:
            break
        page = doc.load_page(pno)
        out.append(_render_page_png_b64(page, dpi=dpi, clip_to_content=clip_to_content))
    return out


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

def _strip_control_chars(s: str) -> str:
    return "".join(ch for ch in s if (ch == "\n" or ch == "\t" or ord(ch) >= 32))

def _blank_invoice_obj() -> Dict[str, Any]:
    return {
        "invoice_date": "",
        "invoice_number": "",
        "gross_invoice_amount": "",
        "invoice_tax": "",
        "invoice_freight": "",
        "po_number": "",
        "po_line_number": "",
        "po_line_amount": "",
        "invoice_description": "",
        "invoice_items": [],
    }


def _normalize_invoice_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return _blank_invoice_obj()

    evidence = obj.get("_evidence")
    norm = _blank_invoice_obj()

    for k in INVOICE_KEYS:
        if k == "invoice_items":
            continue
        v = obj.get(k, "")
        if v is None:
            v = ""
        norm[k] = _strip_control_chars(str(v)).strip()

    norm["po_line_number"] = ""
    norm["po_line_amount"] = ""

    items = obj.get("invoice_items", [])
    out_items: List[Dict[str, str]] = []
    if isinstance(items, list):
        for it in items:
            if not isinstance(it, dict):
                continue
            item_obj: Dict[str, str] = {}
            for ik in ITEM_KEYS:
                vv = it.get(ik, "")
                if vv is None:
                    vv = ""
                item_obj[ik] = _strip_control_chars(str(vv)).strip()

            item_obj["item_number"] = ""
            out_items.append(item_obj)

    filtered: List[Dict[str, str]] = []
    for it in out_items:
        tot = (it.get("item_total") or "").strip()
        tot_n = tot.replace("$", "").replace(",", "").strip()
        if tot_n in ("0", "0.0", "0.00", ""):
            continue
        filtered.append(it)

    norm["invoice_items"] = filtered

    if isinstance(evidence, dict):
        norm["_evidence"] = evidence

    return norm

def _extract_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()

    if t == "The image is not an invoice":
        return None

    fenced = re.findall(r"```json\s*(\{.*?\})\s*```", t, flags=re.IGNORECASE | re.DOTALL)
    for block in fenced:
        try:
            return json.loads(block)
        except Exception:
            continue

    fenced2 = re.findall(r"```\s*(\{.*?\})\s*```", t, flags=re.DOTALL)
    for block in fenced2:
        try:
            return json.loads(block)
        except Exception:
            continue

    i = t.find("{")
    j = t.rfind("}")
    if i != -1 and j != -1 and j > i:
        candidate = t[i : j + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None

    return None

def _build_pages_user_content(page_images_b64: List[str], *, max_pages: Optional[int]) -> List[Dict[str, Any]]:
    imgs = page_images_b64 if max_pages is None else page_images_b64[:max_pages]
    content: List[Dict[str, Any]] = []

    content.append(
        {
            "type": "text",
            "text": (
                "Extract invoice fields from the following PDF pages. "
                "Pages are in order. Use only what you can see. "
                "Follow the system rules exactly."
            ),
        }
    )

    for idx, b64_png in enumerate(imgs, start=1):
        content.append({"type": "text", "text": f"PAGE {idx}:"})
        content.append({"type": "image_url", "image_url": {"url": _to_data_url_png(b64_png)}})

    return content

def _pop_evidence(norm_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ev = norm_obj.get("_evidence")
    if "_evidence" in norm_obj:
        norm_obj.pop("_evidence", None)
    return ev if isinstance(ev, dict) else None


def vision_extract_invoice_json_from_pages(
    page_images_b64: List[str],
    *,
    vision_call: VisionCallable,
    max_pages: Optional[int] = None,
    verify: bool = True,
    return_evidence: bool = False,
    debug_dir: Optional[Path] = None
) -> Tuple[str, Optional[Dict[str, Any]]]:
    if not page_images_b64:
        blank = _blank_invoice_obj()
        out = "```json\n" + json.dumps(blank, ensure_ascii=False, indent=2) + "\n```"
        return out, None

    messages_extract: List[Dict[str, Any]] = [
        {"role": "system", "content": INVOICE_EXTRACT_SYSTEM_PROMPT},
        {"role": "user", "content": _build_pages_user_content(page_images_b64, max_pages=max_pages)},
    ]
    raw_extract = (vision_call(messages_extract) or "").strip()
    if debug_dir:
        (debug_dir / "raw_extract.txt").write_text(raw_extract, encoding="utf-8")


    if raw_extract == "The image is not an invoice":
        return raw_extract, None

    obj = _extract_first_json_obj(raw_extract)
    if obj is None:
        blank = _blank_invoice_obj()
        out = "```json\n" + json.dumps(blank, ensure_ascii=False, indent=2) + "\n```"
        return out, None

    norm = _normalize_invoice_obj(obj)

    if not verify:
        evidence = _pop_evidence(norm)
        out = "```json\n" + json.dumps(norm, ensure_ascii=False, indent=2) + "\n```"
        return out, evidence if return_evidence else None

    candidate_json = json.dumps(norm, ensure_ascii=False, indent=2)
    messages_verify: List[Dict[str, Any]] = [
        {"role": "system", "content": INVOICE_VERIFY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _build_pages_user_content(page_images_b64, max_pages=max_pages)
            + [
                {"type": "text", "text": "CANDIDATE JSON (to verify and correct):"},
                {"type": "text", "text": candidate_json},
            ],
        },
    ]
    raw_verify = (vision_call(messages_verify) or "").strip()
    if debug_dir:
        (debug_dir / "raw_verify.txt").write_text(raw_verify, encoding="utf-8")

    if raw_verify == "The image is not an invoice":
        return raw_verify, None

    obj2 = _extract_first_json_obj(raw_verify)
    if obj2 is None:
        evidence = _pop_evidence(norm)
        out = "```json\n" + json.dumps(norm, ensure_ascii=False, indent=2) + "\n```"
        return out, evidence if return_evidence else None

    norm2 = _normalize_invoice_obj(obj2)
    evidence2 = _pop_evidence(norm2)

    out = "```json\n" + json.dumps(norm2, ensure_ascii=False, indent=2) + "\n```"
    return out, evidence2 if return_evidence else None

def extract_invoice_json_from_pdf_bytes_option_c(
    pdf_bytes: bytes,
    *,
    vision_call: VisionCallable,
    dpi: int = 300,
    clip_to_content: bool = True,
    max_pages: Optional[int] = None,
    verify: bool = True,
    return_evidence: bool = False,
    debug_dir: Optional[Path] = None
) -> Tuple[str, Optional[Dict[str, Any]]]:
    page_imgs = render_all_pdf_pages_as_images_b64(
        pdf_bytes, dpi=dpi, clip_to_content=clip_to_content, max_pages=max_pages
    )
    return vision_extract_invoice_json_from_pages(
        page_imgs,
        vision_call=vision_call,
        max_pages=max_pages,
        verify=verify,
        return_evidence=return_evidence,
        debug_dir=debug_dir,
    )

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

def _load_vision_adapter_from_env() -> Optional[VisionCallable]:
    adapter = os.getenv("PDF_VISION_ADAPTER", "").strip().lower()
    if not adapter:
        return None

    if adapter == "llama32":
        candidates = [("src.scripts.llama32", "llama32"), ("scripts.llama32", "llama32"), ("llama32", "llama32")]
    elif adapter == "pixtral":
        candidates = [("src.scripts.Pixtral", "pixtral"), ("scripts.Pixtral", "pixtral"), ("Pixtral", "pixtral")]
    else:
        return None

    for mod_name, fn_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, fn_name)
            if callable(fn):
                return fn
        except Exception:
            continue

    return None

def main() -> int:
    parser = argparse.ArgumentParser(
        description="PDF text extraction with full-page image fallback for scanned/image-only pages."
    )
    parser.add_argument("pdf", type=str, help="Path to a PDF file to test.")
    parser.add_argument(
        "--mode",
        choices=["render", "ocr", "extract"],
        default="render",
        help="render = render fallback page images; ocr = OCR to text; extract = Option C JSON extraction + verify.",
    )

    parser.add_argument("--dpi", type=int, default=300, help="DPI for rendered page images.")
    parser.add_argument("--no-clip", action="store_true", help="Do not clip to content bbox; render full page area.")
    parser.add_argument("--out-dir", type=str, default="", help="Output directory; default is next to PDF.")
    parser.add_argument("--max-pages", type=int, default=0, help="Limit pages to OCR/extract (0 = no limit).")
    parser.add_argument("--write-text", action="store_true", help="Write extracted/ocr text to out-dir/text.txt.")
    parser.add_argument("--write-json", action="store_true", help="Write extracted invoice JSON to out-dir/invoice.json.")
    parser.add_argument("--write-evidence", action="store_true", help="Write verifier evidence to out-dir/evidence.json.")
    parser.add_argument("--summary-json", action="store_true", help="Also write summary.json to out-dir.")
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

    max_pages = args.max_pages if args.max_pages > 0 else None

    if args.mode == "ocr":
        vision_call = _load_vision_adapter_from_env()
        if not vision_call:
            print("OCR mode requested, but no adapter set or import failed.")
            final_text = res.text
        else:
            final_text = res.text
            if res.fallback_page_images_b64:
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
        return 0

    if args.mode == "extract":
        vision_call = _load_vision_adapter_from_env()
        if not vision_call:
            print("EXTRACT mode requested, but no adapter set or import failed.")
            print(f"\nWrote outputs to: {out_dir}")
            return 0

        invoice_json_text, evidence = extract_invoice_json_from_pdf_bytes_option_c(
            pdf_bytes,
            vision_call=vision_call,
            dpi=args.dpi,
            clip_to_content=(not args.no_clip),
            max_pages=max_pages,
            verify=True,
            return_evidence=args.write_evidence,
            debug_dir=out_dir,
        )
        print("\n[DEBUG] extract() returned:")
        print(f"[DEBUG] invoice_json_text chars: {len(invoice_json_text or '')}")
        print(f"[DEBUG] evidence type: {type(evidence).__name__}")
        if isinstance(evidence, dict):
            print(f"[DEBUG] evidence keys: {list(evidence.keys())[:25]}")
            print(f"[DEBUG] evidence json preview: {json.dumps(evidence, ensure_ascii=False)[:300]}...")
        else:
            print(f"[DEBUG] evidence value preview: {str(evidence)[:300]}")

        if args.write_json:
            (out_dir / "invoice.json").write_text(invoice_json_text, encoding="utf-8")

        if args.write_evidence and isinstance(evidence, dict):
            (out_dir / "evidence.json").write_text(
                json.dumps(evidence, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        print(f"\nWrote outputs to: {out_dir}")
        print("Option C extraction completed.")
        return 0

    if args.write_text:
        (out_dir / "text.txt").write_text(res.text or "", encoding="utf-8")

    print(f"\nWrote outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
