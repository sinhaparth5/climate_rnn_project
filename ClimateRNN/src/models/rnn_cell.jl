# RNN Cell and initialization functions

mutable struct RNNCell
    Wxh::Matrix{Float64}  # Input to hidden weights
    Whh::Matrix{Float64}  # Hidden to hidden weights
    Why::Matrix{Float64}  # Hidden to output weights
    bh::Vector{Float64}   # Hidden bias
    by::Vector{Float64}   # Output bias
    hidden_size::Int64
    dropout_rate::Float64
end

function init_rnn(input_size, hidden_size, output_size; dropout_rate=0.2)
    # Xavier initialization
    scale = sqrt(2.0 / (input_size + hidden_size))
    
    Wxh = randn(hidden_size, input_size) .* scale
    Whh = randn(hidden_size, hidden_size) .* scale
    Why = randn(output_size, hidden_size) .* scale
    
    bh = zeros(hidden_size)
    by = zeros(output_size)
    
    return RNNCell(Wxh, Whh, Why, bh, by, hidden_size, dropout_rate)
end