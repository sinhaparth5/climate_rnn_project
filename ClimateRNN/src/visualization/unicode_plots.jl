# Terminal-based visualization functions

function quick_training_plot(train_losses, val_losses)
    plt = lineplot(1:length(train_losses), train_losses,
        title = "Training History",
        name = "Training Loss",
        xlabel = "Iteration",
        ylabel = "Loss",
        canvas = DotCanvas)
    
    lineplot!(plt, 1:length(val_losses), val_losses,
        name = "Validation Loss")
    
    return plt
end

function quick_prediction_plot(y_true, y_pred; window=100)
    plt = lineplot(1:window, y_true[1:window],
        title = "Predictions vs True Values",
        name = "True Values",
        xlabel = "Time Steps",
        ylabel = "Value",
        canvas = DotCanvas)
    
    lineplot!(plt, 1:window, y_pred[1:window],
        name = "Predictions")
    
    return plt
end

function quick_feature_importance(importance, feature_names)
    plt = barplot(feature_names, importance,
        title = "Feature Importance",
        xlabel = "Features",
        ylabel = "Importance")
    
    return plt
end