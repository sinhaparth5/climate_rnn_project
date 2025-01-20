# Visualization functions using Makie

function detailed_training_plot(train_losses, val_losses)
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
    CairoMakie.scatter!(ax2, dates[1:window], error,
        color = error,
        colormap = :RdBu)
    
    CairoMakie.hlines!(ax2, [0], color = :black, linestyle = :dash)
    
    # Add legends
    CairoMakie.axislegend(ax1)
    
    # Link x-axes
    CairoMakie.linkyaxes!(ax1, ax2)
    
    return fig
end

function detailed_feature_importance(rnn::RNNCell, feature_names)
    importance = vec(sum(abs.(rnn.Wxh), dims=1))
    
    fig = CairoMakie.Figure(resolution = (800, 600))
    ax = CairoMakie.Axis(fig[1, 1],
        xlabel = "Features",
        ylabel = "Absolute Weight Sum",
        title = "Feature Importance")
    
    CairoMakie.barplot!(ax, 1:length(feature_names), importance,
        color = importance,
        colormap = :viridis)
    
    ax.xticks = (1:length(feature_names), feature_names)
    ax.xticklabelrotation = π/4
    
    # Add color bar
    CairoMakie.Colorbar(fig[1, 2], colormap = :viridis,
        label = "Importance")
    
    return fig
end

function correlation_heatmap(df, features)
    cor_matrix = cor(Matrix(df[!, features]))
    
    fig = CairoMakie.Figure(resolution = (800, 800))
    ax = CairoMakie.Axis(fig[1, 1],
        xlabel = "Features",
        ylabel = "Features",
        title = "Feature Correlation Matrix")
    
    CairoMakie.heatmap!(ax, cor_matrix,
        colormap = :RdBu,
        colorrange = (-1, 1))
    
    ax.xticks = (1:length(features), features)
    ax.yticks = (1:length(features), features)
    ax.xticklabelrotation = π/4
    
    CairoMakie.Colorbar(fig[1, 2], colormap = :RdBu,
        colorrange = (-1, 1),
        label = "Correlation")
    
    return fig
end