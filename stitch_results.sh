#!/bin/bash

# for folder in results/NEURAL_SIGMA_UV_*/
# do 
#     ffmpeg -i "${folder}image1_interpolated_neural.jpg" -i "${folder}image4_interpolated_neural.jpeg" -i "${folder}image5_interpolated_neural.jpeg" -filter_complex "[0:v] [1:v] [2:v] hstack=inputs=3" "${folder}horizontal.jpg"
# done

# ffmpeg -i results/NEURAL_SIGMA_UV_0.45_results/horizontal.jpg -i results/NEURAL_SIGMA_UV_0.5_results/horizontal.jpg -i results/NEURAL_SIGMA_UV_0.55_results/horizontal.jpg -i results/NEURAL_SIGMA_UV_0.6_results/horizontal.jpg -i results/NEURAL_SIGMA_UV_0.65_results/horizontal.jpg -i results/NEURAL_SIGMA_UV_0.7_results/horizontal.jpg -i results/NEURAL_SIGMA_UV_0.75_results/horizontal.jpg -i results/NEURAL_SIGMA_UV_0.8_results/horizontal.jpg -filter_complex "[0:v] [1:v] [2:v] [3:v] [4:v] [5:v] [6:v] [7:v] vstack=inputs=8" "results/UV.jpg"

# for folder in results/NEURAL_SIGMA_XY_*/
# do 
#     ffmpeg -i "${folder}image1_interpolated_neural.jpg" -i "${folder}image4_interpolated_neural.jpeg" -i "${folder}image5_interpolated_neural.jpeg" -filter_complex "[0:v] [1:v] [2:v] hstack=inputs=3" "${folder}horizontal.jpg"
# done

# ffmpeg -i results/NEURAL_SIGMA_XY_1.5_results/horizontal.jpg -i results/NEURAL_SIGMA_XY_2.0_results/horizontal.jpg -i results/NEURAL_SIGMA_XY_2.5_results/horizontal.jpg -i results/NEURAL_SIGMA_XY_3.0_results/horizontal.jpg -i results/NEURAL_SIGMA_XY_3.5_results/horizontal.jpg -i results/NEURAL_SIGMA_XY_4.0_results/horizontal.jpg -i results/NEURAL_SIGMA_XY_4.5_results/horizontal.jpg -i results/NEURAL_SIGMA_XY_5.0_results/horizontal.jpg -filter_complex "[0:v] [1:v] [2:v] [3:v] [4:v] [5:v] [6:v] [7:v] vstack=inputs=8" "results/XY.jpg"


# for folder in results/REMOVE_DATA_PROBABILITY_*/
# do 
#     ffmpeg -i "${folder}image1_interpolated_neural.jpg" -i "${folder}image4_interpolated_neural.jpeg" -i "${folder}image5_interpolated_neural.jpeg" -filter_complex "[0:v] [1:v] [2:v] hstack=inputs=3" "${folder}horizontal.jpg"
# done

# ffmpeg -i results/REMOVE_DATA_PROBABILITY_0.0_results/horizontal.jpg -i results/REMOVE_DATA_PROBABILITY_0.1_results/horizontal.jpg -i results/REMOVE_DATA_PROBABILITY_0.2_results/horizontal.jpg -i results/REMOVE_DATA_PROBABILITY_0.3_results/horizontal.jpg -i results/REMOVE_DATA_PROBABILITY_0.4_results/horizontal.jpg -i results/REMOVE_DATA_PROBABILITY_0.5_results/horizontal.jpg -i results/REMOVE_DATA_PROBABILITY_0.6_results/horizontal.jpg -i results/REMOVE_DATA_PROBABILITY_0.7_results/horizontal.jpg -i results/REMOVE_DATA_PROBABILITY_0.8_results/horizontal.jpg -i results/REMOVE_DATA_PROBABILITY_0.9_results/horizontal.jpg -filter_complex "[0:v] [1:v] [2:v] [3:v] [4:v] [5:v] [6:v] [7:v] [8:v] [9:v] vstack=inputs=10" "results/REMOVE.jpg"

for folder in results/NEURAL_SIGMA_UV_*/
do 
    echo "${folder}" >> "results/UV.txt"
    cat "${folder}analysis_output.txt" | grep "NeuralModel - SSIM" >> "results/UV.txt"
    cat "${folder}analysis_output.txt" | grep "NeuralModel - PSNR" >> "results/UV.txt"
    cat "${folder}analysis_output.txt" | grep "NeuralModel - L1" >> "results/UV.txt"
done


for folder in results/NEURAL_SIGMA_XY_*/
do 
    echo "${folder}" >> "results/XY.txt"
    cat "${folder}analysis_output.txt" | grep "NeuralModel - SSIM" >> "results/XY.txt"
    cat "${folder}analysis_output.txt" | grep "NeuralModel - PSNR" >> "results/XY.txt"
    cat "${folder}analysis_output.txt" | grep "NeuralModel - L1" >> "results/XY.txt"
done


for folder in results/REMOVE_DATA_PROBABILITY_*/
do 
    echo "${folder}" >> "results/REMOVE.txt"
    cat "${folder}analysis_output.txt" | grep "NeuralModel - SSIM" >> "results/REMOVE.txt"
    cat "${folder}analysis_output.txt" | grep "NeuralModel - PSNR" >> "results/REMOVE.txt"
    cat "${folder}analysis_output.txt" | grep "NeuralModel - L1" >> "results/REMOVE.txt"
done


