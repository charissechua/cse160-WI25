#define TILE_WIDTH 16
#define KERNEL_SZ 7

__kernel void do_not_remove_this_kernel() {
    int tx = get_local_id(0);
    tx = tx + 1;
}

__kernel void prefn_marker_kernel() {
    int tx = get_local_id(0);
    tx = tx + 1;
}

__kernel void conv_forward_kernel(__global float *y, __global float *x,
    __constant float *k, const int B, const int M, const int C, const int H, 
    const int W, const int K)
{
    /**
    y[b][m][h][w] += x[b][c][h + p][w + q] * k[m][c][p][q]
     */
	//@@ Insert code to implement convolution here
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    //   get_global_id(0) -> w_out
    //   get_global_id(1) -> h_out
    //   get_global_id(2) -> combined index for (b, m)
    int w_out = get_global_id(0);
    int h_out = get_global_id(1);
    int bm = get_global_id(2);
    int m = bm % M;
    int b = bm / M;

    float sum = 0.0f;
    for (int c = 0; c < C; c++) {
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                int h_in = h_out + p;
                int w_in = w_out + q;
                //x[b][c][h + p][w + q]
                // [b] : b * (C * H * W) which batch we are in * size of each batch 
                // [c] : c * (H * W) curr channel * size of each channel 
                // [h + p] : h_in * W current "height" * Width of input offset to find row for which element within the part that is covered by the mask should be multiplied by the k input 
                // [w + q] : w_in is just column
                int x_index = b * (C * H * W) + c * (H * W) + h_in * W + w_in;
                // [m] : m * (C * K * K) = CURR output feature * total number of channels * size of mask 
                // [c] : c * (K * K) = curr channel * size of mask 
                // [p] : curr ROW of the mask with width K 
                // [q] : col of the mask 
                int k_index = m * (C * K * K) + c * (K * K) + p * K + q;
                sum += x[x_index] * k[k_index];
            }
        }
    }
    // Y [b, m, h, w] 
    int y_index = b * (M * H_out * W_out) + m * (H_out * W_out) + h_out * W_out + w_out;
    y[y_index] = sum;
}

