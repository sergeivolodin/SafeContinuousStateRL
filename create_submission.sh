#!/bin/bash
tar cf code_submission.tar *.py requirements.txt README.md *.ipynb output/figures/.placeholder output/*.sh output/figures/*best*.pdf run_*.txt *.sh
echo "code_submission.tar"
