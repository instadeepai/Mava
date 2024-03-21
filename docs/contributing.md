# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Installing Mava for developement

To develop features for Mava, clone the repository and install the core and development dependencies as follows:

```bash
git clone https://github.com/instadeepai/mava.git
cd mava
pip install -e .[dev]
```

## Installing Pre-Commit Hooks

Install the pre-commit hooks:
```bash
pre-commit install
pre-commit install -t commit-msg
```
You can run all the pre-commit hooks on all files as follows:
```bash
pre-commit run --all-files
```

## Naming Conventions
### Branch Names
We name our feature and bugfix branches as follows - `feature/[BRANCH-NAME]`, `bugfix/[BRANCH-NAME]` or `maintenance/[BRANCH-NAME]`. Please ensure `[BRANCH-NAME]` is hyphen delimited.
### Commit Messages
We follow the conventional commits [standard](https://www.conventionalcommits.org/en/v1.0.0/).

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google.com/conduct/).
