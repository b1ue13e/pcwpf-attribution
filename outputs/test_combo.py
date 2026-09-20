import cairosvg

svg = """<svg xmlns="http://www.w3.org/2000/svg" width="500" height="100">
  <rect width="500" height="100" fill="white"/>
  <text x="30" y="55" font-family="'Times New Roman', 'Segoe UI Symbol', 'Cambria Math', serif" font-size="28" font-style="italic">&#8466; &#375; &#955; &#952; &#981; &#945;* <tspan font-family="'Segoe UI Symbol', 'Cambria Math', serif" font-style="normal">&#8712;</tspan> &#8804; &#8800;</text>
</svg>"""
cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to="outputs/test_combo.png")
print("Rendered test_combo.png")
