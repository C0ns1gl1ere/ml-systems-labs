import sys
import pkg_resources

with open("../results/versions.txt", "w") as f:
    f.write(f"Python {sys.version}\n")
    for pkg in pkg_resources.working_set:
        f.write(f"{pkg.key}=={pkg.version}\n")