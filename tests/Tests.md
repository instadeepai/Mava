# Tensorflow System Tests
## How to run
- Build, download packages and test on local environment
    ```bash
    ./bash_scripts/tf_tests.sh
    ```
- Use Docker
    ```
    make build
    make run-tests
    ```
## Structure
```
tests
└── tf
    └── [TEST_FOLDER] - Folder that represents mava component folder, e.g. `wrappers`.
        └── [TEST_FILE] - File that tests a specific file, within a test folder,
              e.g. `wrappers_test.py`.
    └── conftest.py - File that contains test helper functions.
    └── enums.py - File that contains enums used in tests.
    └── mocks.py - File that contains mock classes that are used in tests.
```
