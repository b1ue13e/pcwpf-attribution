import fitz
import cairosvg

# 1. Existing outputs/constraint_lifecycle_taxonomy_assembled.pdf
doc_existing = fitz.open('outputs/constraint_lifecycle_taxonomy_assembled.pdf')
pix_existing = doc_existing[0].get_pixmap(dpi=200)
pix_existing.save('outputs/compare_existing.png')
print('Existing:', pix_existing.width, 'x', pix_existing.height, 'box:', doc_existing[0].mediabox)

# 2. Cairo conversion
cairosvg.svg2pdf(url='outputs/constraint_lifecycle_taxonomy_assembled.svg', write_to='outputs/compare_cairo.pdf')
doc_cairo = fitz.open('outputs/compare_cairo.pdf')
pix_cairo = doc_cairo[0].get_pixmap(dpi=200)
pix_cairo.save('outputs/compare_cairo.png')
print('Cairo:', pix_cairo.width, 'x', pix_cairo.height, 'box:', doc_cairo[0].mediabox)
