# TODO

## Code

* `publish --build` new release so you can `from matplotloom import loom`.
* Is there an issue with `TemporaryDirectory`? Do I need to ensure it doesn't get cleaned up until the end of `__exit__`?
* Possible to turn a bunch of frames into a matplotlib slideshow?
https://github.com/matplotlib/matplotlib/blob/v3.8.3/lib/matplotlib/animation.py#L1320-L1356
https://github.com/matplotlib/matplotlib/blob/v3.8.3/lib/matplotlib/_animation_data.py
* Output writers besides ffmpeg?
* Add tests
* Add CI for many versions of Python (can you test multiple versions of matplotlib too?). Context managers were added in Python 3.5?
* Lots of nice gallery examples. Worth remaking any from the matplotlib gallery?

## Publishing

* https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python
* https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/
* https://github.com/marketplace/actions/pypi-poetry-publish
* https://github.com/marketplace/actions/publish-python-poetry-package
* https://github.com/JRubics/poetry-publish
