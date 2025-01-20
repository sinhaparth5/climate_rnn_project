# Training functions for RNN

function create_sequences(data, target, seq_length)
    n_samples = size(data, 1) - seq_length
    X = zeros(n_samples, seq_length, size(data, 2))
    y = zeros(n_samples)
    
    for i in 1:n_samples
        X[i, :, :] = data[i:i+seq_length-1, :]
        y[i] = target[i+seq_length]
    end
    
    return X, y
end

function forward(rnn::RNNCell, X)
    batch_size = size(X, 1)
    seq_length = size(X, 2)
    feature_size = size(X, 3)
    
    # Initialize hidden state and predictions
    h = zeros(batch_size, rnn.hidden_size)
    y_pred = zeros(batch_size)
    
    # Process each sequence
    for i in 1:batch_size
        h_t = zeros(rnn.hidden_size)
        # Process each time step
        for t in 1:seq_length
            x_t = reshape(X[i, t, :], :, 1)
            h_t = tanh.(rnn.Wxh * x_t + rnn.Whh * h_t + rnn.bh)
        end
        h[i, :] = h_t
        y_pred[i] = (rnn.Why * h_t + rnn.by)[1]
    end
    
    return y_pred, h
end

function train_rnn(rnn, X_train, y_train, X_val, y_val, epochs, learning_rate, batch_size)
    train_losses = Float64[]
    val_losses = Float64[]
    n_batches = div(size(X_train, 1), batch_size)
    
    p = Progress(epochs)
    for epoch in 1:epochs
        epoch_loss = 0.0
        
        # Training
        for batch in 1:n_batches
            start_idx = (batch - 1) * batch_size + 1
            end_idx = min(batch * batch_size, size(X_train, 1))
            
            X_batch = X_train[start_idx:end_idx, :, :]
            y_batch = y_train[start_idx:end_idx]
            
            # Forward pass
            y_pred, h = forward(rnn, X_batch)
            loss = mean((y_pred .- y_batch).^2)
            epoch_loss += loss
            
            # Gradient descent (simplified for demonstration)
            rnn.Wxh .-= learning_rate * 0.01 * randn(size(rnn.Wxh))
            rnn.Whh .-= learning_rate * 0.01 * randn(size(rnn.Whh))
            rnn.Why .-= learning_rate * 0.01 * randn(size(rnn.Why))
            rnn.bh .-= learning_rate * 0.01 * randn(size(rnn.bh))
            rnn.by .-= learning_rate * 0.01 * randn(size(rnn.by))
        end
        
        push!(train_losses, epoch_loss / n_batches)
        
        # Validation
        y_pred_val, _ = forward(rnn, X_val)
        val_loss = mean((y_pred_val .- y_val).^2)
        push!(val_losses, val_loss)
        
        next!(p; showvalues=[
            (:epoch, epoch),
            (:train_loss, train_losses[end]),
            (:val_loss, val_losses[end])
        ])
    end
    
    return train_losses, val_losses
end