import fitz  # pymupdf
import os

ws = "/ai-inventor/aii_data/runs/run_wYelBzy-9k_d/4_gen_paper_repo/_4_assemble_paper/paper/workspace"
pdf_path = os.path.join(ws, "paper.pdf")
out_dir = os.path.join(ws, "pages")
os.makedirs(out_dir, exist_ok=True)

doc = fitz.open(pdf_path)
print(f"Total pages: {len(doc)}")
for i, page in enumerate(doc):
    mat = fitz.Matrix(150/72, 150/72)
    pix = page.get_pixmap(matrix=mat)
    out_path = os.path.join(out_dir, f"page_{i+1:02d}.png")
    pix.save(out_path)
    print(f"Saved page {i+1}: {out_path}")
doc.close()
print("Done.")
