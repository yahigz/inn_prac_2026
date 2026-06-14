$bibtex_use = 2;
$clean_ext = "nav snm";

# Gemini uses fontspec, so compile with LuaLaTeX instead of XeLaTeX.
$pdf_mode = 4;
$lualatex = 'lualatex -interaction=nonstopmode -file-line-error -synctex=1 %O %S';
