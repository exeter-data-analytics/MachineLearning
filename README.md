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
