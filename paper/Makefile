# Build the paper: generated tables/macros from the result files, then pdflatex + bibtex.
# Usage: make -C paper
PY ?= ../.venv/bin/python

main.pdf: main.tex sections/*.tex refs.bib generated/.stamp
	pdflatex -interaction=nonstopmode -halt-on-error main.tex > build.log
	bibtex main >> build.log
	pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build.log
	pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build.log
	@! grep -E "Citation .* undefined|Reference .* undefined|There were undefined" main.log \
	  || (echo "undefined references or citations"; exit 1)

generated/.stamp: make_generated.py
	$(PY) make_generated.py
	touch generated/.stamp

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc build.log

.PHONY: clean
