COVERAGE = coverage
BLACK = black
LINTER = flake8

.PHONY: tests
tests:
	hython tests/tests.py

.PHONY: coverage
coverage:
	hython `which $(COVERAGE)` run --branch --source=soho/python3.7 tests/tests.py
	$(COVERAGE) html

.PHONY: lint
lint:
	$(BLACK) soho/python2.7/*.py
	$(BLACK) soho/python3.7/*.py
	$(BLACK) tests/tests.py
	$(LINTER) soho/python3.7/*.py
	$(LINTER) tests/tests.py

.PHONY: clean
clean:
	/bin/rm -fv ./soho/python2.7/*.pyc
	/bin/rm -fv ./soho/python3.7/*.pyc
	/bin/rm -fvr ./package
	/bin/rm -fvr ./tests/tmp

.PHONY: package
package:
	/bin/rm -frv package
	mkdir -p package/houdini-pbrt-v4
	mkdir package/houdini-pbrt-v4/otls
	hotl -C otls/pbrt.hda package/houdini-pbrt-v4/otls/pbrt.hda
	cp -av soho package/houdini-pbrt-v4/
	cp -av package/houdini-pbrt-v4/soho/python3.7 package/houdini-pbrt-v4/soho/python3.9
	cp -av vop package/houdini-pbrt-v4/
	cp -av houdini-pbrt-v4.json package/
	cd package; zip -r houdini-pbrt-v4.zip \
		houdini-pbrt-v4/otls/pbrt.hda \
		houdini-pbrt-v4/soho \
		houdini-pbrt-v4/vop \
		houdini-pbrt-v4/examples/*.hip \
		houdini-pbrt-v4.json \
		-x *.pyc \
		-9

