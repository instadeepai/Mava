# Tests
## How to run
- Sequential
    ```bash
    pytest -s
    ```
- Parallel
    ```bash
    pytest  -n "$(grep -c ^processor /proc/cpuinfo)" tests
    ```
- Parallel and get code coverage
    ```bash
    pytest --cov-report term-missing --cov=mava  -n "$(grep -c ^processor /proc/cpuinfo)" tests
    ```
- Build, download packages and test
    ```bash
    ./test.sh
    ```
## Structure
```
tests
└── [TEST_FOLDER] - Folder that represents mava component folder, e.g. `wrappers`.
    └── [TEST_FILE] - File that tests a specific file, within a test folder,
       e.g. `wrappers_test.py`.
└── conftest.py - File that contains test helper functions.
└── enums.py - File that contains enums used in tests.
└── mocks.py - File that contains mock classes that are used in tests.
```
