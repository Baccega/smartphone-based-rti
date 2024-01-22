#!/bin/bash

run_data_gathering() {
    local constant_name=$1    # First argument is the constant name
    shift
    local default_value=$1    # Second argument is the default value of the constant
    shift
    model="neural"

    # Loop through each sigma
    for sigma_value in "$@"; do
        # Step 1: Modify the constants.py file
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS uses BSD sed
            sed -i '' "s/^[ \t]*\"$constant_name\": .*/    \"$constant_name\": $sigma_value,/" constants.py
        else
            # Linux uses GNU sed
            sed -i "s/^[ \t]*\"$constant_name\": .*/    \"$constant_name\": $sigma_value,/" constants.py
        fi
        
        # Step 2: Run analysis.py with specific inputs and redirect output to a file
        output_file="analysis_output.txt"
        (echo "3" && echo "6" && echo "y" && echo "y" && echo "y") | python analysis.py > "$output_file"

        # Step 3: Copy the model and output file to a new directory
        new_folder="results/${constant_name}_${sigma_value}_results"
        mkdir -p "$new_folder"
        cp -rf "data/rti/model_$model.pt" "$new_folder/"
        cp -rf data/gaussian_matrix_uv.npz "$new_folder/"
        cp -rf data/gaussian_matrix_xy.npz "$new_folder/"
        cp -rf "$output_file" "$new_folder/"

        # Step 4: Remove the output file
        rm "$output_file"

        # Step 5: Run confront validation
        python confront_validation.py
        cp -rf "image1_interpolated_$model.jpeg" "$new_folder/"
        cp -rf "image2_interpolated_$model.jpeg" "$new_folder/"
        cp -rf "image3_interpolated_$model.jpeg" "$new_folder/"
        cp -rf "image4_interpolated_$model.jpeg" "$new_folder/"
        cp -rf "image5_interpolated_$model.jpeg" "$new_folder/"
        cp -rf image1_ground_truth.jpeg "$new_folder/"
        cp -rf image2_ground_truth.jpeg "$new_folder/"
        cp -rf image3_ground_truth.jpeg "$new_folder/"
        cp -rf image4_ground_truth.jpeg "$new_folder/"
        cp -rf image5_ground_truth.jpeg "$new_folder/"

        # Step 6: Remove the images
        rm "image1_interpolated_$model.jpeg"
        rm "image2_interpolated_$model.jpeg"
        rm "image3_interpolated_$model.jpeg"
        rm "image4_interpolated_$model.jpeg"
        rm "image5_interpolated_$model.jpeg"
        rm image1_ground_truth.jpeg
        rm image2_ground_truth.jpeg
        rm image3_ground_truth.jpeg
        rm image4_ground_truth.jpeg
        rm image5_ground_truth.jpeg

        # Step 7: Modify the constants.py file back to the default value
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS uses BSD sed
            sed -i '' "s/^[ \t]*\"$constant_name\": .*/    \"$constant_name\": $default_value,/" constants.py
        else
            # Linux uses GNU sed
            sed -i "s/^[ \t]*\"$constant_name\": .*/    \"$constant_name\": $default_value,/" constants.py
        fi
    done

    echo "Processing complete for $constant_name."
}

# Define an array of xy_sigma values
# xy_sigmas=("NEURAL_SIGMA_XY" 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0)
xy_sigmas=("NEURAL_SIGMA_XY" 3.0 1.0)
# Define an array of uv_sigma values
# uv_sigmas=("NEURAL_SIGMA_UV" 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85)
uv_sigmas=("NEURAL_SIGMA_UV" 0.6 0.4)
# Define an array of data_removal values    
# data_removal=("REMOVE_DATA_PROBABILITY" 1.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
data_removal=("REMOVE_DATA_PROBABILITY" 0.0 0.0 0.4)

# Run tests for xy_sigmas
run_data_gathering "${xy_sigmas[@]}"

# Run tests for uv_sigmas
run_data_gathering "${uv_sigmas[@]}"

# Run tests for data_removal
run_data_gathering "${data_removal[@]}"
