# Data preprocessing functions

"""
    preprocess_data(input_path::String, output_path::String)

Preprocesses the Delhi climate data by:
1. Converting dates to proper Date format
2. Sorting by date
3. Handling missing values using mean imputation
4. Scaling features using standardization (z-score)

Parameters:
- input_path: Path to the raw CSV file
- output_path: Path where the processed CSV will be saved

Returns:
- DataFrame with preprocessed data
"""
function preprocess_data(input_path::String, output_path::String)
    # Read the data
    df = CSV.read(input_path, DataFrame)
    
    # Convert date strings to Date type if not already Date type
    if !(eltype(df.date) <: Date)
        df.date = Date.(df.date)
    end
    
    # Sort by date
    sort!(df, :date)
    
    # Handle missing values with mean imputation
    for col in names(df)
        if any(ismissing, df[!, col])
            col_mean = mean(skipmissing(df[!, col]))
            df[!, col] = coalesce.(df[!, col], col_mean)
        end
    end
    
    # Scale numerical features
    numerical_cols = ["meantemp", "humidity", "wind_speed", "meanpressure"]
    for col in numerical_cols
        if col in names(df)  # Check if column exists
            df[!, col] = (df[!, col] .- mean(df[!, col])) ./ std(df[!, col])
        end
    end
    
    # Create directory if it doesn't exist
    mkpath(dirname(output_path))
    
    # Save preprocessed data
    CSV.write(output_path, df)
    
    return df
end