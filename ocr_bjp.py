import pytesseract
from pdf2image import convert_from_path
import os

def ocr_pdf_smart(pdf_path, output_path, lang="eng"):
    print(f"Processing {pdf_path}...")
    images = convert_from_path(pdf_path, dpi=300)
    full_text = []
    
    for i, image in enumerate(images):
        print(f"  Page {i+1}/{len(images)}...", end="\r")
        
        # psm 1 = automatic page segmentation with OSD
        # handles columns properly
        custom_config = r'--psm 1'
        text = pytesseract.image_to_string(
            image, 
            lang=lang,
            config=custom_config
        )
        
        if text.strip():
            full_text.append(text)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(full_text))
    
    print(f"\n  Done → {output_path}")

ocr_pdf_smart(
    "Manifestos/bjp.pdf",
    "manifestos_extracted/bjp_raw.txt",
    lang="eng"
)