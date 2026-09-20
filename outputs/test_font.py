import cairosvg
import fitz

fonts = [
    "Times New Roman",
    "Arial",
    "Calibri",
    "Cambria",
    "Georgia"
]

for f in fonts:
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">
      <rect width="200" height="100" fill="white"/>
      <text x="50" y="50" font-family="{f}" font-size="30" font-style="italic">&#375;</text>
    </svg>"""
    out_png = f"outputs/test_{f.replace(' ', '_')}.png"
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=out_png)
    print(f"Generated {out_png}")

# Also test vector path for y_hat
svg_path = """<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">
  <rect width="200" height="100" fill="white"/>
  <g transform="translate(50, 60)">
    <text x="0" y="0" font-family="Times New Roman, serif" font-size="28" font-style="italic">y</text>
    <path d="M-1 -16 L4 -21 L9 -16" fill="none" stroke="#111" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
  </g>
</svg>"""
cairosvg.svg2png(bytestring=svg_path.encode("utf-8"), write_to="outputs/test_yhat_path.png")
print("Generated outputs/test_yhat_path.png")
