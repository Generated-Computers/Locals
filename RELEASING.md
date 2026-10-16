# Releasing Locals

Locals is published to [PyPI](https://pypi.org/p/locals) by the GitHub Actions
workflow [`release.yml`](.github/workflows/release.yml), with trusted publishing:
PyPI trusts that workflow in this repository, so no password or token is kept in
GitHub or anywhere else.

## Once: connect PyPI to this repository

1. Make an account on [pypi.org](https://pypi.org) with two-factor
   authentication, and one on [test.pypi.org](https://test.pypi.org) for trial
   uploads (they are separate sites with separate accounts).
2. On pypi.org, open Your account, then Publishing, and add a pending publisher
   with these values:

   | Field | Value |
   |---|---|
   | PyPI project name | `locals` |
   | Owner | `Generated-Computers` |
   | Repository name | `Locals` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   On test.pypi.org, add the same with the environment name `testpypi`.

   A pending publisher does not reserve the name: the first upload creates the
   project, and until then anyone can register `locals`. Do the first release
   soon after this step.
3. On GitHub, open the repository's Settings, then Environments, and create the
   environments `pypi` and `testpypi`. For `pypi`, add yourself under Required
   reviewers, so that a release waits for your approval before it uploads.

## Each release

1. Set the version in two places, `version` in `pyproject.toml` and
   `__version__` in `locals/__init__.py`, and give its entry in `CHANGELOG.md`
   the date: `## 0.2.0 (2026-10-20)` instead of `## 0.2.0 (unreleased)`.
2. A trial, if you want one: on GitHub, open Actions, then release, then Run
   workflow. It runs the tests, builds the package and uploads it to TestPyPI,
   where its page shows how the README looks on PyPI. To install that build,
   download the wheel alone and install it, so that its dependencies come from
   PyPI and not from TestPyPI:

   ```bash
   python3 -m pip download --no-deps --index-url https://test.pypi.org/simple/ locals==0.2.0 -d dist-test
   uv tool install --force dist-test/locals-0.2.0-py3-none-any.whl
   ```

   TestPyPI keeps every upload: a second trial of the same version uploads
   nothing.
3. Commit, then tag the commit and push the tag:

   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

   The workflow checks that the tag, `pyproject.toml`, `locals/__init__.py` and
   the dated `CHANGELOG.md` entry agree, runs the fast tests on Python 3.11, 3.12
   and 3.13 on a Mac, builds the wheel and the source archive, checks them with
   `twine check --strict`, and, once you approve the `pypi` environment,
   uploads them.
4. Check the release from a clean account or a new Mac user:
   `uv tool install locals`, then `locals --version`.
5. Optionally, on GitHub, make a release from the tag (Releases, then Draft a new
   release) with the version's `CHANGELOG.md` entry as its notes.

A version can be uploaded to PyPI only once. If something is wrong after the
upload, fix it and release the next version (`0.2.1`); a broken version can be
yanked on its PyPI page, which hides it from new installs.

## After the first release

Until `locals` is on PyPI, `install.sh` and the README install from GitHub's
archive of the main branch. Once the first release is up, switch them to PyPI:
the installer then runs `uv tool install locals`, which installs the released
wheel instead of building the main branch, and `uv tool upgrade locals` updates
it. The installer must not point at PyPI before then, since until the first
upload the name `locals` there could belong to anyone.
