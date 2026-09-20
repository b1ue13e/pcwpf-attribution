import cairosvg

test_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="600" height="300">
  <text x="20" y="40" font-family="'DejaVu Math TeX Gyre','FreeSerif','DejaVu Serif',serif" font-size="24">1. DejaVu Math: &#x2112; = &#x2112;_{task} + &#x03bb;&#x2112;_{phys}</text>
  <text x="20" y="80" font-family="'Cambria Math', 'Cambria', serif" font-size="24">2. Cambria Math: &#x2112; = &#x2112;_{task} + &#x03bb;&#x2112;_{phys}</text>
  <text x="20" y="120" font-family="'Segoe UI Symbol', sans-serif" font-size="24">3. Segoe UI Symbol: &#x2112; = &#x2112;_{task} + &#x03bb;&#x2112;_{phys}</text>
  <text x="20" y="160" font-family="'Times New Roman', serif" font-size="24">4. Times New Roman: &#x2112; = &#x2112;_{task} + &#x03bb;&#x2112;_{phys}</text>
  <text x="20" y="200" font-family="'DejaVu Math TeX Gyre','FreeSerif','DejaVu Serif',serif" font-size="24">5. In element: X &#x2208; R^{N&#xd7;T&#xd7;d}</text>
  <text x="20" y="240" font-family="'Cambria Math', serif" font-size="24">6. Cambria in element: X &#x2208; R^{N&#xd7;T&#xd7;d}</text>
</svg>"""

cairosvg.svg2png(bytestring=test_svg.encode('utf-8'), write_to='outputs/font_test.png')
print("Rendered outputs/font_test.png")
