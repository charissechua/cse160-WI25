
__kernel void convolution2D(
    __global int * inputData, __global int * outputData, __constant int * maskData,
    int width, int height, int maskWidth,  int imageChannels, int stride){
    //@@ Insert code to implement matrix multiplication here

    int row_o = get_global_id(1); 
    int col_o = get_global_id(0); 

    //initialize the top left corner of the original image input 
    int row_i = row_o * stride;
    int col_i = col_o * stride;
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
}
