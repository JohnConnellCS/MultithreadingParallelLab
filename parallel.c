/*
 * Name: John Connell
 * UID: 506129117
 */
#include <stdlib.h>
#include <omp.h>
#include "utils.h"
#include "parallel.h"
/*
 * PHASE 1: compute the mean pixel value
 * This code is buggy! Find the bug and speed it up.
 */
void mean_pixel_parallel(const uint8_t img[][NUM_CHANNELS], int num_rows, int
num_cols, double mean[NUM_CHANNELS])
{
 int row, col, ch;
 long count = num_rows * num_cols;
 double pixelSum;

 for (ch = 0; ch < NUM_CHANNELS; ch++)
 {
 pixelSum = 0.0;
#pragma omp parallel for reduction (+:pixelSum)
 for (row = 0; row < num_rows; row++)
 {
 for (col = 0; col < num_cols; col++)
 {
 pixelSum += img[row * num_cols + col][ch];
 }
 }
mean[ch] = pixelSum;
 }
 mean[0] = mean[0] / count;
 mean[1] = mean[1] / count;
 mean[2] = mean[2] / count;
}
/*
 * PHASE 2: convert image to grayscale and record the max grayscale value along
with the number of times it appears
 * This code is NOT buggy, just sequential. Speed it up.
 */
void grayscale_parallel(const uint8_t img[][NUM_CHANNELS], int num_rows, int
num_cols, uint32_t grayscale_img[][NUM_CHANNELS], uint8_t *max_gray, uint32_t
*max_count)
{
 int row, col;
 *max_gray = 0;
 *max_count = 0;
 uint32_t max_noptr = 0;
 uint32_t max_count_noptr = 0;
 uint32_t grayscale_hold = 0;
#pragma omp parallel private (grayscale_hold)
 {
 uint32_t local_max_noptr = 0;
 uint32_t local_max_count_noptr = 0;
#pragma omp for collapse(2)
 for (row = 0; row < num_rows; row++)
 {
 for (col = 0; col < num_cols; col++)
{
 int rowCol = row * num_cols;
 grayscale_hold = img[rowCol + col][0] + img[rowCol + col][1] + img[rowCol
+ col][2];
 grayscale_hold /= 3;
 if(grayscale_hold == local_max_noptr)
 {
 local_max_count_noptr += 3;
 }
 else if (grayscale_hold > local_max_noptr)
 {
 local_max_noptr = grayscale_hold;
 local_max_count_noptr = 3;
 }
 grayscale_img[rowCol + col][0] = grayscale_hold;
 grayscale_img[rowCol + col][1] = grayscale_hold;
 grayscale_img[rowCol + col][2] = grayscale_hold;
 }
 }
#pragma omp critical
{
 if(local_max_noptr == max_noptr)
 {
 max_count_noptr += local_max_count_noptr;
 }
 else if(local_max_noptr > max_noptr)
 {
 max_noptr = local_max_noptr;
 max_count_noptr = local_max_count_noptr;
 }
}
}
 *max_gray = max_noptr;
 *max_count = max_count_noptr;
}
/*
 * PHASE 3: perform convolution on image
 * This code is NOT buggy, just sequential. Speed it up.
 */
void convolution_parallel(const uint8_t padded_img[][NUM_CHANNELS], int num_rows,
int num_cols, const uint32_t kernel[], int kernel_size, uint32_t convolved_img[]
[NUM_CHANNELS])
{
 int row, col, kernel_row, kernel_col;
 int kernel_norm, i;
 int conv_rows, conv_cols;
 // compute kernel normalization factor
 kernel_norm = 0;
#pragma omp parallel for reduction (+:kernel_norm)
 for (i = 0; i < kernel_size * kernel_size; i++)
 {
 kernel_norm += kernel[i];
 }
 // compute dimensions of convolved image
 conv_rows = num_rows - kernel_size + 1;
 conv_cols = num_cols - kernel_size + 1;
 // perform convolution
#pragma omp parallel for collapse (2) private (kernel_row, kernel_col)
 for (row = 0; row < conv_rows; row++)
 {
 for (col = 0; col < conv_cols; col++)
 {
 uint32_t convImgHolderChOne = 0;
uint32_t convImgHolderChTwo = 0;
uint32_t convImgHolderChThree = 0;
 for (kernel_row = 0; kernel_row < kernel_size; kernel_row++)
 {
 int valHolder = (row + kernel_row) * num_cols + col;
 int valHolderTwo = kernel_row * kernel_size;
 for (kernel_col = 0; kernel_col < kernel_size; kernel_col++)
 {
 convImgHolderChOne += padded_img[valHolder + kernel_col][0]
* kernel[valHolderTwo + kernel_col];
convImgHolderChTwo += padded_img[valHolder + kernel_col][1] *
kernel[valHolderTwo + kernel_col];
convImgHolderChThree += padded_img[valHolder + kernel_col][2] *
kernel[valHolderTwo + kernel_col];
 }
 }
 convolved_img[row * conv_cols + col][0] = convImgHolderChOne /
kernel_norm;
convolved_img[row * conv_cols + col][1] = convImgHolderChTwo /
kernel_norm;
convolved_img[row * conv_cols + col][2] = convImgHolderChThree /
kernel_norm;
 }
 }
}
