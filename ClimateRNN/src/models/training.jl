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

function clip_gradient!(grad::Array, threshold::Float64)
    norm_val = norm(grad)
    if norm_val > threshold
        grad .*= (threshold / norm_val)
    end
end

function forward(rnn::RNNCell, X, cache_states=false)
    batch_size = size(X, 1)
    seq_length = size(X, 2)
    
    # Initialize hidden state and predictions
    h = zeros(batch_size, rnn.hidden_size)
    y_pred = zeros(batch_size)
    
    # Store states if needed for backprop
    if cache_states
        h_states = zeros(batch_size, seq_length + 1, rnn.hidden_size)
        x_states = X
    end
    
    # Process each sequence
    for i in 1:batch_size
        h_t = zeros(rnn.hidden_size)
        
        # Process each time step
        for t in 1:seq_length
            x_t = reshape(X[i, t, :], :, 1)
            
            if cache_states
                h_states[i, t, :] = h_t
            end
            
            # Apply dropout during training
            if cache_states  # training mode
                dropout_mask = rand(rnn.hidden_size) .> rnn.dropout_rate
                dropout_mask = dropout_mask ./ (1.0 - rnn.dropout_rate)  # Scale for dropout
                h_t = tanh.(rnn.Wxh * x_t + rnn.Whh * (h_t .* dropout_mask) + rnn.bh)
            else
                h_t = tanh.(rnn.Wxh * x_t + rnn.Whh * h_t + rnn.bh)
            end
        end
        
        if cache_states
            h_states[i, end, :] = h_t
        end
        
        # Final prediction
        y_pred[i] = (rnn.Why * h_t + rnn.by)[1]
    end
    
    if cache_states
        return y_pred, (h_states, x_states)
    else
        return y_pred, nothing
    end
end

function train_rnn(rnn, X_train, y_train, X_val, y_val, epochs, learning_rate, batch_size;
                  momentum=0.9, lr_decay=0.95, grad_clip=5.0)
    train_losses = Float64[]
    val_losses = Float64[]
    n_batches = div(size(X_train, 1), batch_size)
    current_lr = learning_rate
    best_val_loss = Inf
    patience = 0
    
    p = Progress(epochs)
    for epoch in 1:epochs
        epoch_loss = 0.0
        
        # Shuffle training data
        shuffle_idx = randperm(size(X_train, 1))
        X_train = X_train[shuffle_idx, :, :]
        y_train = y_train[shuffle_idx]
        
        # Training
        for batch in 1:n_batches
            start_idx = (batch - 1) * batch_size + 1
            end_idx = min(batch * batch_size, size(X_train, 1))
            
            X_batch = X_train[start_idx:end_idx, :, :]
            y_batch = y_train[start_idx:end_idx]
            
            # Forward pass with cached states
            y_pred, cached_states = forward(rnn, X_batch, true)
            loss = mean((y_pred .- y_batch).^2)
            
            # Skip bad batches
            if isnan(loss) || loss > 1e10
                continue
            end
            
            epoch_loss += loss
            
            # Compute gradients
            h_states, x_states = cached_states
            dWxh = zeros(size(rnn.Wxh))
            dWhh = zeros(size(rnn.Whh))
            dWhy = zeros(size(rnn.Why))
            dbh = zeros(size(rnn.bh))
            dby = zeros(size(rnn.by))
            
            for i in 1:size(X_batch, 1)
                # Output layer gradient
                dy = y_pred[i] - y_batch[i]
                dh = rnn.Why' * [dy]
                dWhy .+= [dy] * h_states[i, end, :]'
                dby .+= [dy]
                
                # Backprop through time
                for t in size(X_batch, 2):-1:1
                    # Gradient through tanh
                    dh = dh .* (1.0 .- h_states[i, t, :].^2)
                    
                    # Accumulate gradients
                    dbh .+= dh
                    dWxh .+= dh * reshape(x_states[i, t, :], 1, :)
                    dWhh .+= dh * h_states[i, t, :]'
                    
                    # Propagate gradient
                    dh = rnn.Whh' * dh
                end
            end
            
            # Clip gradients
            for grad in [dWxh, dWhh, dWhy, dbh, dby]
                clip_gradient!(grad, grad_clip)
            end
            
            # Update with momentum
            rnn.mWxh = momentum * rnn.mWxh - current_lr * dWxh
            rnn.mWhh = momentum * rnn.mWhh - current_lr * dWhh
            rnn.mWhy = momentum * rnn.mWhy - current_lr * dWhy
            rnn.mbh = momentum * rnn.mbh - current_lr * dbh
            rnn.mby = momentum * rnn.mby - current_lr * dby
            
            # Apply updates
            rnn.Wxh .+= rnn.mWxh
            rnn.Whh .+= rnn.mWhh
            rnn.Why .+= rnn.mWhy
            rnn.bh .+= rnn.mbh
            rnn.by .+= rnn.mby
        end
        
        # Calculate average loss
        train_loss = epoch_loss / n_batches
        push!(train_losses, train_loss)
        
        # Validation
        y_pred_val, _ = forward(rnn, X_val, false)
        val_loss = mean((y_pred_val .- y_val).^2)
        push!(val_losses, val_loss)
        
        # Early stopping and learning rate schedule
        if val_loss < best_val_loss
            best_val_loss = val_loss
            patience = 0
        else
            patience += 1
            if patience > 10
                current_lr *= lr_decay
                patience = 0
            end
        end
        
        next!(p; showvalues=[
            (:epoch, epoch),
            (:train_loss, train_loss),
            (:val_loss, val_loss),
            (:learning_rate, current_lr)
        ])
        
        # Stop if learning rate becomes too small
        if current_lr < 1e-6
            println("\nEarly stopping: learning rate too small")
            break
        end
    end
    
    return train_losses, val_losses
end