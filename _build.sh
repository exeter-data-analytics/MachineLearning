#!/bin/sh
#Rscript -e "bookdown::render_book('index.Rmd', 'bookdown::pdf_book')"
Rscript -e "bookdown::render_book('index.Rmd', 'bookdown::gitbook')"

## copy nojekyll file and data
cp .nojekyll _site/ 
zip -r data.zip _data 
cp -r data.zip _site/