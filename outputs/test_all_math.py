import cairosvg

svg = """<svg xmlns="http://www.w3.org/2000/svg" width="400" height="100">
  <rect width="400" height="100" fill="white"/>
  <text x="30" y="50" font-family="'Times New Roman', Times, 'DejaVu Serif', serif" font-size="26" font-style="italic">&#8466; &#375; &#955; &#952; &#981; &#945;* &#8712; &#8804; &#8800;</text>
</svg>"""
cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to="outputs/test_all_math.png")
print("Rendered test_all_math.png")
