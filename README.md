# Climate RNN Project

## Run the project
```bash
using Pkg
Pkg.activate(".")
Pkg.instantiate()

for pkg in ["CSV", "CairoMakie", "ColorSchemes", "DataFrames", "Downloads", 
           "JLD2", "ProgressMeter", "UnicodePlots"]
    Pkg.add(pkg)
end

# Now set up the ClimateRNN package
cd("ClimateRNN")
Pkg.activate(".")
Pkg.instantiate()

# Go back to main project and develop the package
cd("..")
Pkg.activate(".")
Pkg.develop(path="ClimateRNN")
Pkg.resolve()
Pkg.instantiate()
```