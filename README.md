# AdaptiveQuadBench

## Developer Note

- Installation guide (for now, Feb-5-2025), to install original rotorpy to a new conda environment
    ```
    conda env create -f environment.yaml
    conda activate quadbench
    ```
    
- Code structure should follow:
    - Controller
        - L1-Quad
        - MPC
        - Geometric
        - ...
        - Each of the implementation should follow controller/controller_template.py to fit in rotorpy 
    - run_eval.py 
        - run experiments, collect data, etc