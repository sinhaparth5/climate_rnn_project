mutable struct RNNCell
    Wxh::Matrix{Float64}  # Input to hidden weights
    Whh::Matrix{Float64}  # Hidden to hidden weights
    Why::Matrix{Float64}  # Hidden to output weights
    bh::Vector{Float64}   # Hidden bias
    by::Vector{Float64}   # Output bias
    hidden_size::Int64
    dropout_rate::Float64
    # Add momentum terms for better optimization
    mWxh::Matrix{Float64}  # Momentum for Wxh
    mWhh::Matrix{Float64}  # Momentum for Whh
    mWhy::Matrix{Float64}  # Momentum for Why
    mbh::Vector{Float64}   # Momentum for bh
    mby::Vector{Float64}   # Momentum for by
end

function init_rnn(input_size, hidden_size, output_size; dropout_rate=0.2)
    # Xavier initialization with improved scaling
    scale_ih = sqrt(2.0 / (input_size + hidden_size))
    scale_hh = sqrt(2.0 / (hidden_size + hidden_size))
    scale_ho = sqrt(2.0 / (hidden_size + output_size))
    
    Wxh = randn(hidden_size, input_size) .* scale_ih
    Whh = randn(hidden_size, hidden_size) .* scale_hh
    Why = randn(output_size, hidden_size) .* scale_ho
    
    bh = zeros(hidden_size)
    by = zeros(output_size)
    
    # Initialize momentum terms to zero
    mWxh = zeros(size(Wxh))
    mWhh = zeros(size(Whh))
    mWhy = zeros(size(Why))
    mbh = zeros(size(bh))
    mby = zeros(size(by))
    
    return RNNCell(Wxh, Whh, Why, bh, by, hidden_size, dropout_rate,
                  mWxh, mWhh, mWhy, mbh, mby)
end