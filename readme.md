# Explanation of this branch

I made this branch to exclusively contain the framework that I made for my master thesis project, which is contained in `framework_v02/`. It is important to note that I did not end up using it to get the results for the project, as I ran into other problems. Instead I used a script template that I would modify by hand, an example is of it would be `coupled_model_v04_ang_50_U6.py`.

The framework description is explained in chapter 2.4 of the project, although I did not go into much detail. I would add that the inner workings of the coupling framework are in `framework_v02/coupling_framework/` and the specific implementations of the solvers using the framework are done in `framework_v02/solver_wrappers/`. 
The final user experience of the framework is in the `framework_v02/main.py` file.

Again, this code does not run for various reasons. On my machine at the moment it is not running due to problems with directory names. But I hope it can help you as a template for something that can work.

## Sumary of elements

- `beam_corot/` is the elastic model. The aerodynamic code (BEVC) has to be installed on Linux.
- `examples/` contains wind turbine data as "input_/" and the project folders `project_/` (temporal data and results).
- `fromework_v02/` contains the framework code and an example in `main.py`.
- `inertial_forces.py`, `input.py` and `utils.py` contain a variety of functions that the coupled model uses.
