# skeleton_Rmd

This provides a template for R practicals written using the `bookdown` package. The template provides some customised "Task" and "Solution" boxes. To see some examples on how to use this template, download or clone the repository and then compile the project. This can be done on any platform by loading the "skeleton.Rproj" file in RStudio. The "Build Book" button in the "Build" pane can be used to compile the practical. 

After the build is complete, all necessary files will be included in the `docs` folder which is made as part of the build. If you want to link the PDF document within the HTML gitbook document, then you need to compile the PDF first. After compilation, the file `index.html` contains examples on how to use the template.

## Usage

Basically a standard Bookdown template with a few tweaks. New chapters need to be in separate '.Rmd' files, where each file starts with a chapter heading as seen [here](https://bookdown.org/yihui/bookdown/usage.html). In order to use the task and solution blocks in LaTeX, you must input the order of the files into the `_bookdown.yml` file, and the first file must be called `index.Rmd` e.g.

```
rmd_files:
    html: ['index.Rmd', 'ch1.Rmd']
    latex: ['index.Rmd', 'ch1.Rmd', 'ch_appendix.Rmd']
output_dir: "docs"
```

The `latex:` path above ***must*** have `'ch_appendix.Rmd'` as its last entry. This ensures that the appendix is properly formatted for the solutions to the problems.

There are a couple of useful special blocks. A `task` block, and a `solution` block. These can be used as e.g.

````
```{task}
Here is a task written in **markdown**.
```
````

To see how this renders, please compile the template and open "docs/index.html".

You can include chunks within the `task` chunk, but you need to use double backticks *within* the chunk, and leave carriage returns around the internal chunk e.g.

````

```{task}

``{r}
x <- 2 + 2
x
``

```

````

Again, to see how this renders, please compile the template.

Be careful to have suitable carriage returns around e.g. `enumerate` or `itemize` environments inside the chunk also. For example:

````

```{task}
Here is a list:
1. item 1
2. item 2
```

```` 

will not render nicely. But

````

```{task}
Here is a list:

1. item 1
2. item 2

```

```` 

will.

The `solution` chunk works in the same way, and the numbers will follow the previous `task` chunk (so you can set tasks without solutions) e.g.

````

```{task}
Add 2 and 2 together
```

```{solution}

``{r}
2 + 2
``

```

````

### Tabbed boxed environments

Originally developed to put base R and `tidyverse` solutions side-by-side, using a `multCode = T` option to the solution box. Here the two tabs are sepearetd by four consecutive hashes: `####`, and the `titles` option gives the tab titles (these can be set globally if preferred) e.g.



````

```{task}
Filter the `iris` data by `Species == "setosa"` and summarise.
```

```{solution, multCode = T, titles = c("Base R", "tidyverse")}

``{r}
## base R solution
summary(iris[iris$Species == "setosa", ])
``

####

``{r}
## tidyverse solution
iris %>% 
    filter(Species == "setosa") %>%
    summary()
``
    
```

````

will typeset to:

<div class="panel panel-default"><div class="panel-heading"> Task </div><div class="panel-body"> 
Filter the `iris` data by `Species == "setosa"` and summarise. </div></div>

<button id="displayTextunnamed-chunk-11" onclick="javascript:toggle('unnamed-chunk-11');">Show Solution</button>

<div id="toggleTextunnamed-chunk-11" style="display: none"><div class="panel panel-default"><div class="panel-heading panel-heading1"> Solution </div><div class="panel-body"><div class="tab"><button class="tablinksunnamed-chunk-11 active" onclick="javascript:openCode(event, 'option1unnamed-chunk-11', 'unnamed-chunk-11');">Base R</button><button class="tablinksunnamed-chunk-11" onclick="javascript:openCode(event, 'option2unnamed-chunk-11', 'unnamed-chunk-11');">tidyverse</button></div><div id="option1unnamed-chunk-11" class="tabcontentunnamed-chunk-11">

```r
## base R solution
summary(iris[iris$Species == "setosa", ])
```

```
##   Sepal.Length    Sepal.Width     Petal.Length    Petal.Width   
##  Min.   :4.300   Min.   :2.300   Min.   :1.000   Min.   :0.100  
##  1st Qu.:4.800   1st Qu.:3.200   1st Qu.:1.400   1st Qu.:0.200  
##  Median :5.000   Median :3.400   Median :1.500   Median :0.200  
##  Mean   :5.006   Mean   :3.428   Mean   :1.462   Mean   :0.246  
##  3rd Qu.:5.200   3rd Qu.:3.675   3rd Qu.:1.575   3rd Qu.:0.300  
##  Max.   :5.800   Max.   :4.400   Max.   :1.900   Max.   :0.600  
##        Species  
##  setosa    :50  
##  versicolor: 0  
##  virginica : 0  
##                 
##                 
## 
```
</div><div id="option2unnamed-chunk-11" class="tabcontentunnamed-chunk-11">

```r
## tidyverse solution
iris %>% 
    filter(Species == "setosa") %>%
    summary()
```

```
##   Sepal.Length    Sepal.Width     Petal.Length    Petal.Width   
##  Min.   :4.300   Min.   :2.300   Min.   :1.000   Min.   :0.100  
##  1st Qu.:4.800   1st Qu.:3.200   1st Qu.:1.400   1st Qu.:0.200  
##  Median :5.000   Median :3.400   Median :1.500   Median :0.200  
##  Mean   :5.006   Mean   :3.428   Mean   :1.462   Mean   :0.246  
##  3rd Qu.:5.200   3rd Qu.:3.675   3rd Qu.:1.575   3rd Qu.:0.300  
##  Max.   :5.800   Max.   :4.400   Max.   :1.900   Max.   :0.600  
##        Species  
##  setosa    :50  
##  versicolor: 0  
##  virginica : 0  
##                 
##                 
## 
```
</div><script> javascript:hide('option2unnamed-chunk-11') </script></div></div></div>

Note that there is also a `multCode` chunk that does not link to task and solution boxes e.g.

<div class="tab"><button class="tablinksunnamed-chunk-12 active" onclick="javascript:openCode(event, 'option1unnamed-chunk-12', 'unnamed-chunk-12');">Option 1</button><button class="tablinksunnamed-chunk-12" onclick="javascript:openCode(event, 'option2unnamed-chunk-12', 'unnamed-chunk-12');">Option 2</button></div><div id="option1unnamed-chunk-12" class="tabcontentunnamed-chunk-12">

Two options: 

* Option 1
</div><div id="option2unnamed-chunk-12" class="tabcontentunnamed-chunk-12">

Two options:
    
* Option 2
</div><script> javascript:hide('option2unnamed-chunk-12') </script>

The `titles` option can be set as before.

