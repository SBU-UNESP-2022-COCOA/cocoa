from cobaya.yaml import yaml_load_file
from cobaya.run import run



info_from_yaml =yaml_load_file("./projects/des_y3/EXAMPLE_MCMC91.input.yaml")

updated_info_minimizer, minimizer = run(info_from_yaml, minimize=True)
# To get the maximum-a-posteriori:
print(minimizer.products()["minimum"])