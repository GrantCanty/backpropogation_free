# Continual Learning Research

This repository is the workspace for a new ground-up investigation of
continuous learning, predictive representations, and subquadratic alternatives
to exact recursive least squares.

The completed CPAM/no-backprop project is preserved as a self-contained,
reproducible archive under [`archives/cpam`](archives/cpam). It includes its
source, configurations, tests, result artifacts, paper outline, literature
review, and final positive and negative findings.

The archived OS-ELM implementation remains an important baseline for future
work. New code should not import CPAM internals accidentally; reusable pieces
should be brought back explicitly with their own tests and documented reason.
