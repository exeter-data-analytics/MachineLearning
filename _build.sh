#!/bin/sh
Rscript -e "bookdown::render_book('index.Rmd', 'bookdown::pdf_book')"
Rscript -e "bookdown::render_book('index.Rmd', 'bookdown::gitbook')"

## copy nojekyll file and data
cp .nojekyll _site/ 
#cp -r _data _site/
#cp -r dataFiles.zip _site/