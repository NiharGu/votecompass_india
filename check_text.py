import pdfplumber
import os

FOLDER = "Manifestos"

files = {
    "BJP":      "bjp.pdf",
    "INC":      "inc.pdf",
    "TMC":      "tmc.pdf",
    "NCP(SP)":  "ncp(sp).pdf",
    "NCP(AP)":  "ncp(ap).pdf",
    "SP":       "sp.pdf",
    "DMK":      "dmk.pdf",
}

for party, filename in files.items():
    print(f"\n{'='*50}")
    print(f"PARTY: {party}")
    print('='*50)
    
    filepath = os.path.join(FOLDER, filename)
    
    with pdfplumber.open(filepath) as pdf:
        total = len(pdf.pages)
        
        # Check pages 1, 2, 3, and middle — cover is often a blank/image
        test_pages = [1, 2, 3, total // 2]
        
        for pg_num in test_pages:
            if pg_num >= total:
                continue
            text = pdf.pages[pg_num].extract_text()
            char_count = len(text.strip()) if text else 0
            
            if char_count > 100:
                print(f"✅ Page {pg_num+1} has text ({char_count} chars)")
                print(f"   Preview: {text[:200].strip()}")
                print(f"   → REST OF PDF LIKELY TEXT-NATIVE, skip OCR")
                break
            else:
                print(f"⚠️  Page {pg_num+1} empty ({char_count} chars)")
        else:
            print(f"❌ All sampled pages empty — genuinely scanned, needs OCR")