#!/bin/bash

# GENERATE IMAGES

# rm -rf "results/NEURAL_SIGMA_UV_0.45_results/horizontal-heading.jpg"
# ffmpeg \
#     -i "results/NEURAL_SIGMA_UV_0.45_results/image1_interpolated_neural.jpeg" \
#     -i "results/NEURAL_SIGMA_UV_0.45_results/image1_ground_truth.jpeg" \
#     -filter_complex " \
#     [0:v]pad=iw:ih+50:0:50:color=white[v0p]; \
#     [v0p]drawtext=text='Interpolated':fontcolor=black:fontsize=24:x=(w-text_w)/2:y=20[v0]; \
#     [1:v]pad=iw:ih+50:0:50:color=white[v1p]; \
#     [v1p]drawtext=text='Ground Truth':fontcolor=black:fontsize=24:x=(w-text_w)/2:y=20[v1]; \
#     [v0][v1]hstack=inputs=2" \
#     "results/NEURAL_SIGMA_UV_0.45_results/horizontal-heading.jpg"

# for folder in results/NEURAL_SIGMA_UV_*/
# do 
#     rm -rf "${folder}horizontal.jpg"
#     ffmpeg -i "${folder}image1_interpolated_neural.jpeg" -i "${folder}image1_ground_truth.jpeg" -filter_complex "[0:v] [1:v] hstack=inputs=2" "${folder}horizontal.jpg"

# rm -rf "results/UV.jpg"

# ffmpeg \
# -i results/NEURAL_SIGMA_UV_0.45_results/horizontal-heading.jpg \
# -i results/NEURAL_SIGMA_UV_0.5_results/horizontal.jpg \
# -i results/NEURAL_SIGMA_UV_0.55_results/horizontal.jpg \
# -i results/NEURAL_SIGMA_UV_0.6_results/horizontal.jpg \
# -i results/NEURAL_SIGMA_UV_0.65_results/horizontal.jpg \
# -i results/NEURAL_SIGMA_UV_0.7_results/horizontal.jpg \
# -i results/NEURAL_SIGMA_UV_0.75_results/horizontal.jpg \
# -i results/NEURAL_SIGMA_UV_0.8_results/horizontal.jpg \
# -filter_complex "\
# [0:v]pad=iw+100:ih:100:0:color=white,drawtext=text='0.45':fontcolor=black:fontsize=24:x=25:y=(h-text_h)/2[v0]; \
# [1:v]pad=iw+100:ih:100:0:color=white,drawtext=text='0.5':fontcolor=black:fontsize=24:x=25:y=(h-text_h)/2[v1]; \
# [2:v]pad=iw+100:ih:100:0:color=white,drawtext=text='0.55':fontcolor=red:fontsize=24:x=25:y=(h-text_h)/2[v2]; \
# [3:v]pad=iw+100:ih:100:0:color=white,drawtext=text='0.6':fontcolor=black:fontsize=24:x=25:y=(h-text_h)/2[v3]; \
# [4:v]pad=iw+100:ih:100:0:color=white,drawtext=text='0.65':fontcolor=black:fontsize=24:x=25:y=(h-text_h)/2[v4]; \
# [5:v]pad=iw+100:ih:100:0:color=white,drawtext=text='0.7':fontcolor=black:fontsize=24:x=25:y=(h-text_h)/2[v5]; \
# [6:v]pad=iw+100:ih:100:0:color=white,drawtext=text='0.75':fontcolor=black:fontsize=24:x=25:y=(h-text_h)/2[v6]; \
# [7:v]pad=iw+100:ih:100:0:color=white,drawtext=text='0.8':fontcolor=black:fontsize=24:x=25:y=(h-text_h)/2[v7]; \
# [v0][v1][v2][v3][v4][v5][v6][v7]vstack=inputs=8" \
# "results/UV.jpg"

# rm -rf "results/NEURAL_SIGMA_XY_1.5_results/horizontal-heading.jpg"
# ffmpeg \
#     -i "results/NEURAL_SIGMA_XY_1.5_results/image1_interpolated_neural.jpeg" \
#     -i "results/NEURAL_SIGMA_XY_1.5_results/image1_ground_truth.jpeg" \
#     -filter_complex " \
#     [0:v]pad=iw:ih+50:0:50:color=white[v0p]; \
#     [v0p]drawtext=text='Interpolated':fontcolor=black:fontsize=24:x=(w-text_w)/2:y=20[v0]; \
#     [1:v]pad=iw:ih+50:0:50:color=white[v1p]; \
#     [v1p]drawtext=text='Ground Truth':fontcolor=black:fontsize=24:x=(w-text_w)/2:y=20[v1]; \
#     [v0][v1]hstack=inputs=2" \
#     "results/NEURAL_SIGMA_XY_1.5_results/horizontal-heading.jpg"

# for folder in results/NEURAL_SIGMA_XY_*/
# do 
#     ffmpeg -i "${folder}image1_interpolated_neural.jpeg" -i "${folder}image1_ground_truth.jpeg" -filter_complex "[0:v] [1:v] hstack=inputs=2" "${folder}horizontal.jpg"
# done
# rm -rf "results/XY.jpg"

# ffmpeg \
# -i results/NEURAL_SIGMA_XY_1.5_results/horizontal-heading.jpg \
# -i results/NEURAL_SIGMA_XY_2.0_results/horizontal.jpg \
# -i results/NEURAL_SIGMA_XY_2.5_results/horizontal.jpg \
# -i results/NEURAL_SIGMA_XY_3.0_results/horizontal.jpg \
# -i results/NEURAL_SIGMA_XY_3.5_results/horizontal.jpg \
# -i results/NEURAL_SIGMA_XY_4.0_results/horizontal.jpg \
# -i results/NEURAL_SIGMA_XY_4.5_results/horizontal.jpg \
# -i results/NEURAL_SIGMA_XY_5.0_results/horizontal.jpg \
# -filter_complex "\
# [0:v]pad=iw+100:ih:100:0:color=white,drawtext=text='1.5':fontcolor=black:fontsize=24:x=25:y=(h-text_h)/2[v0]; \
# [1:v]pad=iw+100:ih:100:0:color=white,drawtext=text='2.0':fontcolor=black:fontsize=24:x=25:y=(h-text_h)/2[v1]; \
# [2:v]pad=iw+100:ih:100:0:color=white,drawtext=text='2.5':fontcolor=black:fontsize=24:x=25:y=(h-text_h)/2[v2]; \
# [3:v]pad=iw+100:ih:100:0:color=white,drawtext=text='3.0':fontcolor=red:fontsize=24:x=25:y=(h-text_h)/2[v3]; \
# [4:v]pad=iw+100:ih:100:0:color=white,drawtext=text='3.5':fontcolor=black:fontsize=24:x=25:y=(h-text_h)/2[v4]; \
# [5:v]pad=iw+100:ih:100:0:color=white,drawtext=text='4.0':fontcolor=black:fontsize=24:x=25:y=(h-text_h)/2[v5]; \
# [6:v]pad=iw+100:ih:100:0:color=white,drawtext=text='4.5':fontcolor=black:fontsize=24:x=25:y=(h-text_h)/2[v6]; \
# [7:v]pad=iw+100:ih:100:0:color=white,drawtext=text='5.0':fontcolor=black:fontsize=24:x=25:y=(h-text_h)/2[v7]; \
# [v0][v1][v2][v3][v4][v5][v6][v7]vstack=inputs=8" \
# "results/XY.jpg"


rm -rf "results/REMOVE_DATA_PROBABILITY_0.0_results/horizontal-heading.jpg"
ffmpeg \
    -i "results/REMOVE_DATA_PROBABILITY_0.0_results/image1_interpolated_neural.jpeg" \
    -i "results/REMOVE_DATA_PROBABILITY_0.0_results/image1_ground_truth.jpeg" \
    -filter_complex " \
    [0:v]pad=iw:ih+50:0:50:color=white[v0p]; \
    [v0p]drawtext=text='Interpolated':fontcolor=black:fontsize=24:x=(w-text_w)/2:y=20[v0]; \
    [1:v]pad=iw:ih+50:0:50:color=white[v1p]; \
    [v1p]drawtext=text='Ground Truth':fontcolor=black:fontsize=24:x=(w-text_w)/2:y=20[v1]; \
    [v0][v1]hstack=inputs=2" \
    "results/REMOVE_DATA_PROBABILITY_0.0_results/horizontal-heading.jpg"

for folder in results/REMOVE_DATA_PROBABILITY_*/
do 
    ffmpeg -i "${folder}image1_interpolated_neural.jpeg" -i "${folder}image1_ground_truth.jpeg" -filter_complex "[0:v] [1:v] hstack=inputs=2" "${folder}horizontal.jpg"
done
rm -rf "results/REMOVE.jpg"

ffmpeg \
-i results/REMOVE_DATA_PROBABILITY_0.0_results/horizontal-heading.jpg \
-i results/REMOVE_DATA_PROBABILITY_0.1_results/horizontal.jpg \
-i results/REMOVE_DATA_PROBABILITY_0.2_results/horizontal.jpg \
-i results/REMOVE_DATA_PROBABILITY_0.3_results/horizontal.jpg \
-i results/REMOVE_DATA_PROBABILITY_0.4_results/horizontal.jpg \
-i results/REMOVE_DATA_PROBABILITY_0.5_results/horizontal.jpg \
-i results/REMOVE_DATA_PROBABILITY_0.6_results/horizontal.jpg \
-i results/REMOVE_DATA_PROBABILITY_0.7_results/horizontal.jpg \
-i results/REMOVE_DATA_PROBABILITY_0.8_results/horizontal.jpg \
-i results/REMOVE_DATA_PROBABILITY_0.9_results/horizontal.jpg \
-filter_complex "\
[0:v]pad=iw+100:ih:100:0:color=white,drawtext=text='0.0':fontcolor=black:fontsize=24:x=25:y=(h-text_h)/2[v0]; \
[1:v]pad=iw+100:ih:100:0:color=white,drawtext=text='0.1':fontcolor=black:fontsize=24:x=25:y=(h-text_h)/2[v1]; \
[2:v]pad=iw+100:ih:100:0:color=white,drawtext=text='0.2':fontcolor=black:fontsize=24:x=25:y=(h-text_h)/2[v2]; \
[3:v]pad=iw+100:ih:100:0:color=white,drawtext=text='0.3':fontcolor=black:fontsize=24:x=25:y=(h-text_h)/2[v3]; \
[4:v]pad=iw+100:ih:100:0:color=white,drawtext=text='0.4':fontcolor=black:fontsize=24:x=25:y=(h-text_h)/2[v4]; \
[5:v]pad=iw+100:ih:100:0:color=white,drawtext=text='0.5':fontcolor=black:fontsize=24:x=25:y=(h-text_h)/2[v5]; \
[6:v]pad=iw+100:ih:100:0:color=white,drawtext=text='0.6':fontcolor=black:fontsize=24:x=25:y=(h-text_h)/2[v6]; \
[7:v]pad=iw+100:ih:100:0:color=white,drawtext=text='0.7':fontcolor=black:fontsize=24:x=25:y=(h-text_h)/2[v7]; \
[8:v]pad=iw+100:ih:100:0:color=white,drawtext=text='0.8':fontcolor=black:fontsize=24:x=25:y=(h-text_h)/2[v8]; \
[9:v]pad=iw+100:ih:100:0:color=white,drawtext=text='0.9':fontcolor=black:fontsize=24:x=25:y=(h-text_h)/2[v9]; \
[v0][v1][v2][v3][v4][v5][v6][v7][v8][v9]vstack=inputs=10" \
"results/REMOVE.jpg"

# GET DATA FROM FILES

# for folder in results/NEURAL_SIGMA_UV_*/
# do 
#     echo "${folder}" >> "results/UV.txt"
#     cat "${folder}analysis_output.txt" | grep "NeuralModel - SSIM" >> "results/UV.txt"
#     cat "${folder}analysis_output.txt" | grep "NeuralModel - PSNR" >> "results/UV.txt"
#     cat "${folder}analysis_output.txt" | grep "NeuralModel - L1" >> "results/UV.txt"
# done


# for folder in results/NEURAL_SIGMA_XY_*/
# do 
#     echo "${folder}" >> "results/XY.txt"
#     cat "${folder}analysis_output.txt" | grep "NeuralModel - SSIM" >> "results/XY.txt"
#     cat "${folder}analysis_output.txt" | grep "NeuralModel - PSNR" >> "results/XY.txt"
#     cat "${folder}analysis_output.txt" | grep "NeuralModel - L1" >> "results/XY.txt"
# done


# for folder in results/REMOVE_DATA_PROBABILITY_*/
# do 
#     echo "${folder}" >> "results/REMOVE.txt"
#     cat "${folder}analysis_output.txt" | grep "NeuralModel - SSIM" >> "results/REMOVE.txt"
#     cat "${folder}analysis_output.txt" | grep "NeuralModel - PSNR" >> "results/REMOVE.txt"
#     cat "${folder}analysis_output.txt" | grep "NeuralModel - L1" >> "results/REMOVE.txt"
# done


