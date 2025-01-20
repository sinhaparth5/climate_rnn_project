module ClimateRNN

# Import all required packages
using Downloads
using DataFrames
using CSV
using Statistics
using Dates
using LinearAlgebra
using Random
using ProgressMeter
using JLD2
using UnicodePlots
using CairoMakie
using ColorSchemes

# Include all source files
include(joinpath(@__DIR__, "data", "preprocess.jl"))
include(joinpath(@__DIR__, "models", "rnn_cell.jl"))
include(joinpath(@__DIR__, "models", "training.jl"))
include(joinpath(@__DIR__, "visualization", "makie_plots.jl"))
include(joinpath(@__DIR__, "visualization", "unicode_plots.jl"))

# Export functions
export download_climate_data, preprocess_data
export RNNCell, init_rnn, create_sequences, train_rnn, forward
export quick_training_plot, quick_prediction_plot, quick_feature_importance
export detailed_training_plot, detailed_prediction_plot, detailed_feature_importance
export correlation_heatmap

end