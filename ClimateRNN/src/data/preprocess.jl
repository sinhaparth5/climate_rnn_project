# Data preprocessing functions

"""
    generate_sample_data()

Generates synthetic daily climate data for testing.
Returns a path to the generated CSV file.
"""
function generate_sample_data()
    n_days = 1000
    
    # Generate dates
    start_date = Date(2020, 1, 1)
    dates = start_date .+ Day.(0:n_days-1)
    
    # Generate synthetic data with seasonal patterns and some noise
    t = range(0, n_days-1, length=n_days)
    meantemp = 20 .+ 10 .* sin.(2π .* t ./ 365) .+ 2 .* randn(n_days)  # Temperature with seasonal variation
    humidity = 60 .+ 20 .* sin.(2π .* t ./ 365 .+ π/4) .+ 5 .* randn(n_days)  # Humidity with seasonal variation
    wind_speed = 10 .+ 5 .* randn(n_days)  # Random wind speed
    meanpressure = 1013 .+ 10 .* sin.(2π .* t ./ 365 .+ π/2) .+ 2 .* randn(n_days)  # Pressure with seasonal variation
    
    # Create DataFrame
    df = DataFrame(
        date = dates,
        meantemp = meantemp,
        humidity = clamp.(humidity, 0, 100),  # Humidity between 0-100%
        wind_speed = abs.(wind_speed),  # Non-negative wind speed
        meanpressure = meanpressure
    )
    
    # Save to CSV
    output_path = joinpath(@__DIR__, "..", "..", "..", "data", "raw", "climate_data.csv")
    mkpath(dirname(output_path))
    CSV.write(output_path, df)
    
    return output_path
end

"""
    download_climate_data()

Generates synthetic climate data for testing since we can't access external data.
"""
function download_climate_data()
    return generate_sample_data()
end

"""
    preprocess_data(input_path::String, output_path::String)

Preprocesses the climate data by:
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