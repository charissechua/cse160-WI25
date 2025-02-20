
__kernel void convolution2D(
    __global int * inputData, __global int * outputData, __constant int * maskData,
    int width, int height, int maskWidth,  int imageChannels, int stride){
    //@@ Insert code to implement matrix multiplication here

    int row_o = get_global_id(1); 
    int col_o = get_global_id(0); 

    //initialize the top left corner of the original image input 
    int row_i = row_o; 
    int col_i = col_o; 
    int sum = 0;
    int i, j, mask_i, mask_j, image_val, mask_val;

    int output_width = (width - (maskWidth - 1))/stride;
    int output_height = (height - (maskWidth - 1))/stride;

    for (int k = 0; k < imageChannels;k++){
        sum = 0;
        for (i = row_i, mask_i = 0 ; i < row_i + maskWidth; i ++, mask_i++){
            for (j = col_i, mask_j = 0; j < col_i + maskWidth; j++, mask_j++){
                mask_val = maskData[mask_i * maskWidth + mask_j]; 
                image_val = inputData[(i * width + j) * imageChannels + k];
                sum += mask_val * image_val;
            }
        }
        outputData [((row_o * output_width) + col_o) * imageChannels + k] = sum;
    }
//     int maskRadius = maskWidth / 2;
//     int row_l = get_local_id(1); 
//     int col_l = get_local_id(0); 

//     //row/col we are mapping to in output 
//     int row_o = get_global_id(1); //+ row_l; // todo need to adjust for tiling 
//     int col_o = get_global_id(0); //+ col_l;

//     //declaring outside of for loop for speed
//     int image_val, mask_val;
//     int i, y, x, sum;
//    // printf("row = %d , col = %d \n", row_o, col_o); 
//     int row_i = row_o; 
//     int col_i = col_o;
//     int output_width = (width - (maskWidth - 1))/stride;
//     int output_height = (height - (maskWidth - 1))/stride;
//     if (row_i >= 0 && (row_o + maskRadius) < height){
//         if (col_i >= 0 && (col_o + maskRadius) < width){
//             for (i = 0; i < imageChannels; i ++){
//                 sum = 0; 
//                 for (y = 0; y < maskWidth; y++){
//                     for (x = 0; x < maskWidth; x++){
//                         image_val = inputData[((row_o + y) * width + x + col_o) * imageChannels + i]; 
//                         mask_val = maskData[(y) * maskWidth + x]; 
//                         sum += image_val * mask_val;
//                     }
//                 }
//                 //printf("the sum for (row, col) (%d, %d) is %d \n", row_o, col_o, sum);
//                 outputData[((row_o * output_width) + col_o) * imageChannels + i] = sum;
//             }
//         }
//     }
}
