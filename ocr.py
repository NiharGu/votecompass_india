import pytesseract
from pdf2image import convert_from_path
import os

FOLDER = "Manifestos"
OUTPUT_FOLDER = "manifestos_extracted"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def ocr_pdf(pdf_path, output_path, lang="hin"):
    print(f"\nProcessing: {pdf_path}")
    images = convert_from_path(pdf_path, dpi=300)
    print(f"Pages: {len(images)}")
    
    full_text = []
    for i, image in enumerate(images):
        print(f"  OCR page {i+1}/{len(images)}...", end="\r")
        text = pytesseract.image_to_string(image, lang=lang)
        if text.strip():
            full_text.append(text)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(full_text))
    
    print(f"\n  Saved to {output_path}")
    
    # Quality check
    print(f"\n--- QUALITY CHECK: {os.path.basename(pdf_path)} (first 500 chars) ---")
    with open(output_path, "r", encoding="utf-8") as f:
        print(f.read()[:500])
    print("---")

# SP — full Hindi
ocr_pdf(
    os.path.join(FOLDER, "sp.pdf"),
    os.path.join(OUTPUT_FOLDER, "sp_raw.txt"),
    lang="hin"
)

# NCP(AP) — check if Hindi or English or mixed
# Try eng first, if garbled we switch to hin
ocr_pdf(
    os.path.join(FOLDER, "ncp(ap).pdf"),
    os.path.join(OUTPUT_FOLDER, "ncp_ap_raw.txt"),
    lang="eng"
)