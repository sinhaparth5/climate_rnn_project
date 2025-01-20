using Pkg
Pkg.activate(dirname(@__DIR__))

using ClimateRNN
using DataFrames
using CSV
using Statistics
using Dates
using JLD2
using UnicodePlots
using CairoMakie
CairoMakie.activate!()

# Load and preprocess data
raw_data_path = joinpath(@__DIR__, "..", "data", "raw", "DailyDelhiClimate.csv")
processed_data_path = joinpath(@__DIR__, "..", "data", "processed", "preprocessed_data.csv")

# Preprocess the data
df = preprocess_data(raw_data_path, processed_data_path)

# Quick terminal visualization of data distribution
println("Data Distribution:")
println(UnicodePlots.boxplot(df.meantemp, title="Temperature Distribution"))

# Prepare sequences for training
features = ["meantemp", "humidity", "wind_speed", "meanpressure"]
X_data = Matrix(df[!, features])
y_data = df.meantemp

# Create sequences for RNN training
seq_length = 30  # Back to original sequence length
X, y = create_sequences(X_data, y_data, seq_length)

# Split data into training and validation sets
n_samples = size(X, 1)
train_size = Int(floor(0.8 * n_samples))
X_train = X[1:train_size, :, :]
y_train = y[1:train_size]
X_val = X[train_size+1:end, :, :]
y_val = y[train_size+1:end]

# Initialize RNN with smaller hidden size
input_size = length(features)
hidden_size = 32  # Back to original hidden size
output_size = 1
rnn = init_rnn(input_size, hidden_size, output_size, dropout_rate=0.1)  # Lower dropout

# Training parameters
epochs = 100
learning_rate = 0.001  # Lower learning rate
batch_size = 32

# Train the model
println("\nTraining the RNN model...")
train_losses, val_losses = train_rnn(rnn, X_train, y_train, X_val, y_val, 
                                   epochs, learning_rate, batch_size,
                                   momentum=0.9, lr_decay=0.99,  # Slower decay
                                   grad_clip=1.0)  # Conservative gradient clipping

# Generate predictions
println("\nGenerating predictions...")
y_pred_train, _ = forward(rnn, X_train)
y_pred_val, _ = forward(rnn, X_val)

# Calculate MSE
train_mse = mean((y_pred_train .- y_train).^2)
val_mse = mean((y_pred_val .- y_val).^2)
println("\nTraining MSE: ", train_mse)
println("Validation MSE: ", val_mse)

# Create visualizations
println("\nCreating detailed visualizations...")
results_dir = joinpath(@__DIR__, "..", "results", "figures")
mkpath(results_dir)

# Training history plot
fig1 = detailed_training_plot(train_losses, val_losses)
save(joinpath(results_dir, "training_history.png"), fig1)

# Prediction plot
dates_val = df.date[train_size+seq_length+1:end]
fig2 = detailed_prediction_plot(y_val, vec(y_pred_val), dates_val)
save(joinpath(results_dir, "predictions.png"), fig2)

# Feature importance plot
fig3 = detailed_feature_importance(rnn, features)
save(joinpath(results_dir, "feature_importance.png"), fig3)

# Correlation heatmap
fig4 = correlation_heatmap(df, features)
save(joinpath(results_dir, "correlation_heatmap.png"), fig4)

println("\nAnalysis complete! Results saved in the 'results' directory.")