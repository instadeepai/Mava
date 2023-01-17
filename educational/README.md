# MAVA Educational

## How to run
Inside of the `educational` folder, run:
### Docker
```
make build && make run example=[]
```
, e.g. `make build && make run example=systems/template.py`.

### Python virtual env
- venv:
    ```
    python -m venv mava_edu
    source mava_edu/bin/activate
    ```
- OR conda:
    ```
    conda create -n mava_edu python=3.9
    conda activate mava_edu
    ``` 
Then install requirements:
```
pip install --upgrade pip setuptools
pip install -r requirements.txt 
```