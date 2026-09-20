import os
import re

def test_highlights():
    path = "01_论文投稿材料/submission_package/HIGHLIGHTS_APPLIED_ENERGY.md"
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    bullets = []
    for line in lines:
        line = line.strip()
        if line.startswith("- ") and "chars]" in line:
            b = line[2:line.rfind("[")].strip()
            bullets.append(b)
            
    print(f"Total bullets found: {len(bullets)}")
    for i, b in enumerate(bullets, 1):
        print(f"Bullet {i} ({len(b)} chars): {b}")
        assert len(b) <= 85, f"Bullet {i} exceeds 85 chars! ({len(b)} chars)"
    print("All highlights passed <= 85 character limit!")

if __name__ == "__main__":
    test_highlights()
