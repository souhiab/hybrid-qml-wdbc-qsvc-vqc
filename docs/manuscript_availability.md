# Manuscript: Data and Code Availability statements

> The manuscript LaTeX source is not part of this repository. Paste the
> following statements into the manuscript (e.g. `main.tex`) to address the
> editor's request for an accessible link. Ensure the preamble loads
> `hyperref` (or at least the `url` package) so that `\url{}` compiles, e.g.
> `\usepackage{hyperref}`.

```latex
\section*{Data availability}
The Wisconsin Diagnostic Breast Cancer (WDBC) dataset used in this study is
publicly available from the UCI Machine Learning Repository and through the
\texttt{sklearn.datasets.load\_breast\_cancer} utility. No private or
personally identifiable clinical data were used in this study. The processed
experimental outputs and reproducibility materials are available in the
accompanying public GitHub repository:
\url{https://github.com/souhiab/hybrid-qml-wdbc-qsvc-vqc}.

\section*{Code availability}
The source code used to perform preprocessing, feature selection, classical
and quantum machine learning experiments, evaluation, and figure/table
generation is publicly available at:
\url{https://github.com/souhiab/hybrid-qml-wdbc-qsvc-vqc}.
```

If the manuscript already uses a single combined **"Data and code
availability"** section, update that section in place rather than adding the
two separate sections above, to avoid duplication.

Repository link to verify in the manuscript:
`https://github.com/souhiab/hybrid-qml-wdbc-qsvc-vqc`
