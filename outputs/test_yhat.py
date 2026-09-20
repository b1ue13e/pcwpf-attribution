import cairosvg

svg = """<svg xmlns="http://www.w3.org/2000/svg" width="600" height="300">
<text x="20" y="50" font-family="DejaVu Math TeX Gyre" font-size="28">1. DejaVu Math: ŷ</text>
<text x="20" y="100" font-family="Times New Roman" font-size="28">2. Times New Roman: ŷ</text>
<text x="20" y="150" font-family="DejaVu Serif" font-size="28">3. DejaVu Serif: ŷ</text>
<text x="20" y="200" font-family="Cambria Math" font-size="28">4. Cambria Math: ŷ</text>
<text x="20" y="250" font-family="FreeSerif" font-size="28">5. FreeSerif: ŷ</text>
</svg>"""

cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to="outputs/test_yhat.png")
print("Rendered test_yhat.png successfully")
