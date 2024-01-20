#!/bin/bash

# Define an array of sigma values
sigmas=(1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0)

# Loop through each sigma
for sigma in "${sigmas[@]}"
do
    # Step 1: Modify the constants.py file
    sed -i "s/NEURAL_SIGMA_XY: .*/NEURAL_SIGMA_XY: $sigma,/" constants.py

    # Step 2: Run analysis.py with specific inputs and redirect output to a file
    output_file="analysis_output_sigma_${sigma}.txt"
    (echo "3" && echo "6" && echo "y" && echo "y" && echo "y") | python analysis.py > "$output_file"

    # Step 3: Copy the model and output file to a new directory
    new_folder="results/sigma_${sigma}_results"
    mkdir -p "$new_folder"
    cp data/rti/neural_model.pt "$new_folder/"
    cp data/rti/gaussian_matrix_uv.npz "$new_folder/"
    cp data/rti/gaussian_matrix_xy.npz "$new_folder/"
    cp "$output_file" "$new_folder/"

    # Step 4: Remove the output file
    rm "$output_file"

    # Step 5: Run confront validation
    python confront_validation.py
    cp image1_interpolated.jpeg "$new_folder/"
    cp image2_interpolated.jpeg "$new_folder/"
    cp image3_interpolated.jpeg "$new_folder/"
    cp image4_interpolated.jpeg "$new_folder/"
    cp image5_interpolated.jpeg "$new_folder/"
    cp image1_ground_truth.jpeg "$new_folder/"
    cp image2_ground_truth.jpeg "$new_folder/"
    cp image3_ground_truth.jpeg "$new_folder/"
    cp image4_ground_truth.jpeg "$new_folder/"
    cp image5_ground_truth.jpeg "$new_folder/"

    # Step 6: Remove the images
    rm image1_interpolated.jpeg
    rm image2_interpolated.jpeg
    rm image3_interpolated.jpeg
    rm image4_interpolated.jpeg
    rm image5_interpolated.jpeg
    rm image1_ground_truth.jpeg
    rm image2_ground_truth.jpeg
    rm image3_ground_truth.jpeg
    rm image4_ground_truth.jpeg
    rm image5_ground_truth.jpeg

done

echo "Processing complete."
