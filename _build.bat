Rscript.exe -e "bookdown::render_book('index.Rmd', 'bookdown::gitbook')"
Rscript.exe -e "bookdown::render_book('index.Rmd', 'bookdown::pdf_book')"

copy .nojekyll docs/
