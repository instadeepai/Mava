## How to add docs

For each new page in the docs, you will need to add `.md` file to the `docs` folder and then add a route to this page in `mkdocs.yml` (under `nav`).

To build this locally, you will need to download the docs requirements, `pip install -r docs/requirements.txt`.

### Adding a page:
#### 1.  Auto generated from docstrings.
We use [mkdocstrings](https://github.com/mkdocstrings/mkdocstrings) to do this. Point the docs page to your python file/class, e.g. `::: mava.wrappers.debugging_envs` to point to the [debugging_env](https://github.com/instadeepai/Mava/blob/develop/mava/wrappers/debugging_envs.py) file.

#### 2. Reference an existing `.md` file.
We use the [snippets add on](https://facelessuser.github.io/pymdown-extensions/extensions/snippets/) to reference existing `.md` files. Point the docs page to the relevant `.md` file, e.g. `--8<-- "README.md"` to point to the base `README.md` file in Mava.

#### 3. Manual Doc Page
You can manually create a docs page.
