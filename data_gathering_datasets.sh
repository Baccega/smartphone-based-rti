#!/bin/bash

output_file="analysis_output.txt"
(echo "3" && echo "6" && echo "y" && echo "y" && echo "y") | python analysis.py > "$output_file"

new_folder="results/datasets/rti"
mkdir -p "$new_folder"
cp -rf "data/rti/model_neural.pt" "$new_folder/"
cp -rf data/gaussian_matrix_uv.npz "$new_folder/"
cp -rf data/gaussian_matrix_xy.npz "$new_folder/"
cp -rf "$output_file" "$new_folder/"

# Step 4: Remove the output file
rm "$output_file"

# Step 5: Run confront validation
python confront_validation.py
cp -rf "image1_interpolated_neural.jpeg" "$new_folder/"
cp -rf "image2_interpolated_neural.jpeg" "$new_folder/"
cp -rf "image3_interpolated_neural.jpeg" "$new_folder/"
cp -rf "image4_interpolated_neural.jpeg" "$new_folder/"
cp -rf "image5_interpolated_neural.jpeg" "$new_folder/"
cp -rf image1_ground_truth.jpeg "$new_folder/"
cp -rf image2_ground_truth.jpeg "$new_folder/"
cp -rf image3_ground_truth.jpeg "$new_folder/"
cp -rf image4_ground_truth.jpeg "$new_folder/"
cp -rf image5_ground_truth.jpeg "$new_folder/"

# Step 6: Remove the images
rm "image1_interpolated_neural.jpeg"
rm "image2_interpolated_neural.jpeg"
rm "image3_interpolated_neural.jpeg"
rm "image4_interpolated_neural.jpeg"
rm "image5_interpolated_neural.jpeg"
rm image1_ground_truth.jpeg
rm image2_ground_truth.jpeg
rm image3_ground_truth.jpeg
rm image4_ground_truth.jpeg
rm image5_ground_truth.jpeg


# SYNTH


output_file="analysis_output.txt"
(echo "2" && echo "6" && echo "y" && echo "y" && echo "y") | python analysis.py > "$output_file"

new_folder="results/datasets/synth"
mkdir -p "$new_folder"
cp -rf "data/synth/model_neural.pt" "$new_folder/"
cp -rf data/gaussian_matrix_uv.npz "$new_folder/"
cp -rf data/gaussian_matrix_xy.npz "$new_folder/"
cp -rf "$output_file" "$new_folder/"

# Step 4: Remove the output file
rm "$output_file"

# Step 5: Run confront validation
python confront_validation.py
cp -rf "image1_interpolated_neural.jpeg" "$new_folder/"
cp -rf "image2_interpolated_neural.jpeg" "$new_folder/"
cp -rf "image3_interpolated_neural.jpeg" "$new_folder/"
cp -rf "image4_interpolated_neural.jpeg" "$new_folder/"
cp -rf "image5_interpolated_neural.jpeg" "$new_folder/"
cp -rf image1_ground_truth.jpeg "$new_folder/"
cp -rf image2_ground_truth.jpeg "$new_folder/"
cp -rf image3_ground_truth.jpeg "$new_folder/"
cp -rf image4_ground_truth.jpeg "$new_folder/"
cp -rf image5_ground_truth.jpeg "$new_folder/"

# Step 6: Remove the images
rm "image1_interpolated_neural.jpeg"
rm "image2_interpolated_neural.jpeg"
rm "image3_interpolated_neural.jpeg"
rm "image4_interpolated_neural.jpeg"
rm "image5_interpolated_neural.jpeg"
rm image1_ground_truth.jpeg
rm image2_ground_truth.jpeg
rm image3_ground_truth.jpeg
rm image4_ground_truth.jpeg
rm image5_ground_truth.jpeg

