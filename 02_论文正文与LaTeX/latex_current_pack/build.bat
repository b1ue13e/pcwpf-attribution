@echo off
"C:\Users\32110\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" -interaction=nonstopmode -file-line-error main.tex
bibtex main
"C:\Users\32110\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" -interaction=nonstopmode -file-line-error main.tex
"C:\Users\32110\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" -interaction=nonstopmode -file-line-error main.tex
