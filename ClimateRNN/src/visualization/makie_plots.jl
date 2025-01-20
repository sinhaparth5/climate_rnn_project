# Visualization functions using Makie

function detailed_training_plot(train_losses, val_losses)
    # Remove any NaN values
    train_losses = filter(!isnan, train_losses)
    val_losses = filter(!isnan, val_losses)
    
    fig = CairoMakie.Figure(resolution = (800, 600))
    ax = CairoMakie.Axis(fig[1, 1],
        xlabel = "Iteration",
        ylabel = "Loss",
        title = "Training History")
    
    CairoMakie.lines!(ax, 1:length(train_losses), train_losses,
        label = "Training Loss",
        color = :blue)
    CairoMakie.lines!(ax, 1:length(val_losses), val_losses,
        label = "Validation Loss",
        color = :red)
    
    CairoMakie.axislegend(ax)
    return fig
end

function detailed_prediction_plot(y_true, y_pred, dates; window=100)
    # Clean up any NaN values
    valid_idx = .!isnan.(y_true) .& .!isnan.(y_pred)
    y_true = y_true[valid_idx]
    y_pred = y_pred[valid_idx]
    dates = dates[valid_idx]

    window = min(window, length(y_true))

    fig = CairoMakie.Figure(resolution = (1000, 600))

    # Main prediction plot
    ax1 = CairoMakie.Axis(fig[1, 1],
        xlabel = "Date",
        ylabel = "Value",
        title = "Predictions vs True Values")

    CairoMakie.lines!(ax1, dates[1:window], y_true[1:window],
        label = "True Values",
        color = :blue)
    CairoMakie.lines!(ax1, dates[1:window], y_pred[1:window],
        label = "Predictions",
        color = :red)

    # Error plot
    ax2 = CairoMakie.Axis(fig[2, 1],
        xlabel = "Date",
        ylabel = "Error")

    error = y_pred[1:window] .- y_true[1:window]

    # Clean error values for plotting
    valid_errors = .!isnan.(error)
    error_dates = dates[1:window][valid_errors]
    clean_errors = error[valid_errors]

    # Replace NaN with 0
    clean_errors = replace(clean_errors, NaN => 0.0) 

    if !isempty(clean_errors)
        CairoMakie.scatter!(ax2, error_dates, clean_errors,
            color = clean_errors,
            colormap = :RdBu,
            colorrange = (-maximum(abs.(clean_errors)), maximum(abs.(clean_errors))))
    end

    CairoMakie.hlines!(ax2, [0], color = :black, linestyle = :dash)

    # Add legends
    CairoMakie.axislegend(ax1)

    # Link x-axes
    CairoMakie.linkyaxes!(ax1, ax2)

    return fig
end

function detailed_feature_importance(rnn::RNNCell, feature_names)
    importance = vec(sum(abs.(rnn.Wxh), dims=1))
    
    # Make sure there are no NaN values
    importance = replace(importance, NaN => 0.0)
    
    fig = CairoMakie.Figure(resolution = (800, 600))
    ax = CairoMakie.Axis(fig[1, 1],
        xlabel = "Features",
        ylabel = "Absolute Weight Sum",
        title = "Feature Importance")
    
    CairoMakie.barplot!(ax, 1:length(feature_names), importance,
        color = importance,
        colormap = :viridis,
        colorrange = (minimum(importance), maximum(importance)))
    
    ax.xticks = (1:length(feature_names), feature_names)
    ax.xticklabelrotation = π/4
    
    # Add color bar
    CairoMakie.Colorbar(fig[1, 2], colormap = :viridis,
        colorrange = (minimum(importance), maximum(importance)),
        label = "Importance")
    
    return fig
end

function correlation_heatmap(df, features)
    # Calculate correlation matrix
    data = Matrix(df[!, features])
    n_features = length(features)
    cor_matrix = zeros(n_features, n_features)
    
    # Manually calculate correlations to handle NaN values
    for i in 1:n_features
        for j in 1:n_features
            valid_idx = .!isnan.(data[:, i]) .& .!isnan.(data[:, j])
            if any(valid_idx)
                cor_matrix[i, j] = cor(data[valid_idx, i], data[valid_idx, j])
            end
        end
    end
    
    # Replace any remaining NaN values with 0
    cor_matrix = replace(cor_matrix, NaN => 0.0)
    
    # Create the figure
    fig = CairoMakie.Figure(resolution = (800, 800))
    ax = CairoMakie.Axis(fig[1, 1],
        xlabel = "Features",
        ylabel = "Features",
        title = "Feature Correlation Matrix")
    
    # Create heatmap
    hm = CairoMakie.heatmap!(ax, cor_matrix,
        colormap = :RdBu,
        colorrange = (-1, 1))
    
    # Set axis labels
    ax.xticks = (1:n_features, features)
    ax.yticks = (1:n_features, features)
    ax.xticklabelrotation = π/4
    
    # Add colorbar
    CairoMakie.Colorbar(fig[1, 2], 
        hm,  # Link colorbar to heatmap
        colormap = :RdBu,
        label = "Correlation")
    
    return fig
end