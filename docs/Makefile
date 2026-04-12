# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = QCANT
SOURCEDIR     = .
BUILDDIR      = _build

# Prefer building with the *active* environment.
# Sphinx's "make mode" (-M) requires the `sphinx-build` executable.
ifdef CONDA_PREFIX
SPHINXBUILD = $(CONDA_PREFIX)/bin/sphinx-build
else ifdef VIRTUAL_ENV
SPHINXBUILD = $(VIRTUAL_ENV)/bin/sphinx-build
else ifneq (,$(wildcard ../.venv/bin/sphinx-build))
SPHINXBUILD = ../.venv/bin/sphinx-build
else
SPHINXBUILD = sphinx-build
endif

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: htmlq

# Quiet HTML build (suppresses most Sphinx output). Use this instead of
# `make html -q` (which can be interpreted by `make` as a target named "-q").
htmlq:
	@$(SPHINXBUILD) -q -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)