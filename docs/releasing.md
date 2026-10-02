# Release procedure

The target is `commcode` version `0.2.0`. Version `0.1.0` was a local prototype;
this numbering does not imply that a prior version was published to PyPI.
A successful local build does not establish that a public release exists.

## Publisher registration

Register a pending GitHub Trusted Publisher separately on TestPyPI and PyPI.
This requires the owner's authenticated account; do not paste credentials into
issues, chat, or the repository. The exact fields are:

| Field | TestPyPI | PyPI |
| --- | --- | --- |
| Project name | `commcode` | `commcode` |
| Owner | `haochengw372-hash` | `haochengw372-hash` |
| Repository | `commcode` | `commcode` |
| Workflow filename | `release.yml` | `release.yml` |
| Environment | `testpypi` | `pypi` |

Register at <https://test.pypi.org/manage/account/publishing/> and
<https://pypi.org/manage/account/publishing/>. Pending publisher registration does
not reserve a name and must be reconfirmed when publishing.

## Verification and publication

1. Run package tests and Ruff; build an sdist and wheel in an isolated build environment.
2. Run `twine check` and `scripts/audit_release.py` over the actual archive contents.
   The audit uses a positive file allowlist and checks for local paths and common
   secret formats; human review still applies to source and documentation.
3. Install the wheel into a fresh environment outside the source checkout, run the
   synthetic example and CLI help, and confirm the installed version and path.
4. Create the tagged candidate from this verified tree. Dispatch `release.yml`
   with `target=testpypi` and its exact version; the workflow publishes and installs
   the public artifact in a fresh verification job.
5. After TestPyPI verification succeeds for that commit/version, dispatch
   `target=pypi`. The workflow requires the matching successful TestPyPI run and
   reuses its exact distribution artifacts rather than building new files.
6. Verify PyPI JSON metadata, published SHA-256 hashes, and installation from the
   public index in a new environment. Only then describe the package as released.

Publishing uses short-lived GitHub Actions OIDC credentials. `id-token: write`
is scoped to publish jobs; normal pushes and pull requests cannot publish.
Environments should restrict publishing to the intended release refs. The initial
repository push runs tests only.

See the authoritative instructions for
[creating a project with Trusted Publishing](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
and the [PyPA publishing action](https://github.com/pypa/gh-action-pypi-publish).
