import cairosvg

fonts = ["Cambria Math", "Segoe UI Symbol", "Arial", "DejaVu Sans", "Calibri"]
for f in fonts:
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="200" height="80">
      <rect width="200" height="80" fill="white"/>
      <text x="30" y="50" font-family="{f}" font-size="28">&#8712;</text>
    </svg>"""
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=f"outputs/test_in_{f.replace(' ', '_')}.png")
print("Rendered in tests")
