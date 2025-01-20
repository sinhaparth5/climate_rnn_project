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
raw_data_path = joinpath(@__DIR__, "..", "data", "raw", "climate_data.csv")
processed_data_path = joinpath(@__DIR__, "..", "data", "processed", "preprocessed_data.csv")

# Download data if it doesn't exist
if !isfile(raw_data_path)
    download_climate_data()
end

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
seq_length = 30  # Use 30 days of data to predict the next day
X, y = create_sequences(X_data, y_data, seq_length)

# Split data into training and validation sets
n_samples = size(X, 1)
train_size = Int(floor(0.8 * n_samples))
X_train = X[1:train_size, :, :]
y_train = y[1:train_size]
X_val = X[train_size+1:end, :, :]
y_val = y[train_size+1:end]

# Initialize RNN
input_size = length(features)
hidden_size = 32
output_size = 1
rnn = init_rnn(input_size, hidden_size, output_size, dropout_rate=0.2)

# Training parameters
epochs = 100
learning_rate = 0.01
batch_size = 32

# Train the model
println("\nTraining the RNN model...")
train_losses, val_losses = train_rnn(rnn, X_train, y_train, X_val, y_val, 
                                   epochs, learning_rate, batch_size)

# Plot training progress
println("\nTraining Progress:")
quick_plot = quick_training_plot(train_losses, val_losses)
println(quick_plot)

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

# Quick terminal visualization of predictions
println("\nQuick Prediction View:")
println(quick_prediction_plot(y_val, vec(y_pred_val)))

# Save trained model
println("\nSaving trained model...")
model_path = joinpath(@__DIR__, "..", "results", "models", "trained_model.jld2")
mkpath(dirname(model_path))
@save model_path rnn

println("\nAnalysis complete! Results saved in the 'results' directory.")