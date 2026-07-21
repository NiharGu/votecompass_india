import subprocess
import os

FOLDER = "Manifestos"
OUTPUT = "manifestos_extracted"
os.makedirs(OUTPUT, exist_ok=True)

parties = {
    "bjp":   "bjp.pdf",
    "inc":   "inc.pdf",
    "cpim":  "cpi-m.pdf",
    "tmc":   "tmc.pdf",
    "ncpsp": "ncp(sp).pdf",
    "dmk":   "dmk.pdf",
}

for party, filename in parties.items():
    input_path  = os.path.join(FOLDER, filename)
    output_path = os.path.join(OUTPUT, f"{party}_raw.txt")
    
    print(f"Extracting {party}...", end=" ")
    
    result = subprocess.run(
        ["pdftotext", input_path, output_path],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        size = os.path.getsize(output_path)
        print(f"✅ {size:,} bytes → {output_path}")
    else:
        print(f"❌ Error: {result.stderr}")

print("\nDone. All files in manifestos_extracted/")