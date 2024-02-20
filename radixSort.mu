#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <utility>

#include <musa.h>
#include <musa_runtime_api.h>

// https://web.archive.org/web/20221012085306/https://vgc.poly.edu/~csilva/papers/cgf.pdf
//
// Input data:   [1 2 0 3] [0 1 1 0] [3 3 3 2] [1 2 2 0] [2 0 0 2]
// Local PfxSums...
//     ... of 0: [    0  ] [0     1] [       ] [      0] [  0 1  ]
//     ... of 1: [0      ] [  0 1  ] [       ] [0      ] [       ]
//     ... of 2: [  0    ] [       ] [      0] [  0 1  ] [0     1]
//     ... of 3: [      0] [       ] [0 1 2  ] [       ] [       ]
//     Combined: [0 0 0 0] [0 0 1 1] [0 1 2 0] [0 0 1 0] [0 0 1 1]
//
// Occurrences of bit patterns per block
//     ... 0:     1         2         0         1         2  |  6 total
//     ... 1:     1         2         0         1         0  |  4 total
//     ... 2:     1         0         1         2         2  |  6 total
//     ... 3:     1         0         3         0         0  |  4 total
//
// Global Pfxsums:
//                        Shift             Scan              +offset
//     ... 0: [1 2 0 1 2] ----> [0 1 2 0 1] ----> [0 1 3 3 4] -- +0 -->  [0   1  3  3  4]
//     ... 1: [1 2 0 1 0] ----> [0 1 2 0 1] ----> [0 1 3 3 4] -- +6 -->  [6   7  9  9 10]
//     ... 2: [1 0 1 2 2] ----> [0 1 0 1 2] ----> [0 1 1 2 4] -- +10 ->  [10 11 11 12 14]
//     ... 3: [1 0 3 0 0] ----> [0 1 0 3 0] ----> [0 1 1 4 4] -- +16 ->  [16 17 17 20 20]
//
// Apply Local PfxSums to Global Pfxsums to get shuffle table
//
// Input data    [1  2  0  3] [0  1  1  0] [ 3  3  3  2] [1  2  2  0] [2  0  0  2]
// Local Pfxsums [0  0  0  0] [0  0  1  1] [ 0  1  2  0] [0  0  1  0] [0  0  1  1]
// Offsets
//     ... 0:    [      0   ] [1        1] [           ] [         3] [   4  4   ]
//     ... 1:    [6         ] [   7  7   ] [           ] [9         ] [          ]
//     ... 2:    [  10      ] [          ] [         11] [  12 12   ] [14      14]
//     ... 3:    [        16] [          ] [17 17 17   ] [          ] [          ]
// Dest Idx :    [6 10  0 16] [1  7  8  2] [17 18 19 11] [9 12 13  3] [14 4  5 15]
//
// Output:
//      Idx      0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
//      Elt      0  0  0  0  0  0  1  1  1  1  2  2  2  2  2  2  3  3  3  3
//

// Using only 1 thread block
template<typename T>
__device__ void SingleBlockBlellochScan(T* in, T* out, int N) {
    assert(N <= blockDim.x);
    T* ping = in, *pong = out;
    int dist = 1;
    int iter = 0;
    __syncthreads();
    bool need_swap = true;
    while (dist < N) {
        need_swap = !need_swap;
        for (int i=threadIdx.x; i<N; i+=blockDim.x) {
            if (i >= dist && i < N) {
                pong[i] = ping[i] + ping[i - dist];
            } else if (i < dist) {
                pong[i] = ping[i];
            }
        }
        T* tmp = ping; ping = pong; pong = tmp;
        dist *= 2;
        iter++;
        __syncthreads();
    }

    if (need_swap) {
        for (int i=threadIdx.x; i<N; i+=blockDim.x) {
            out[i] = ping[i];
        }
    }
}

__global__ void SingleBlockBlellochScanTest(short* d_in, short* d_out, int N) {
    extern __shared__ char shmem[];
    short* ping = (short*)&(shmem[0]);
    short* pong = (ping + blockDim.x);
    // Populate input data
    ping[threadIdx.x] = d_in[threadIdx.x];
    SingleBlockBlellochScan(ping, pong, N);
    d_out[threadIdx.x] = pong[threadIdx.x];
}

//  Input Elt     1 2 0 3     0 1 1 0     3 3 3 2     1 2 2 0     2 0 0 2
//  Local PfxSum  0 0 0 0     0 0 1 1     0 1 2 0
__device__ void BlockLevelSort(int* in, int lb, int ub, int shift_right, int* dbg_out) {
    extern __shared__ char shmem[];
    const int N = ub-lb+1;
    short* sh_flags        = (short*)(&shmem[0]);
    short* sh_flags_cumsum = sh_flags + blockDim.x;
    short* sh_local_pfxsum = sh_flags_cumsum + blockDim.x;

    dbg_out[threadIdx.x] = 0;

    for (int bmask=0; bmask<4; bmask++) {
        // Local prefix sum
        sh_flags[threadIdx.x] = 0;
        __syncthreads();
        if (lb+threadIdx.x <= ub) {
            int elt = (in[lb+threadIdx.x] >> shift_right) & 3;
            if (bmask == elt) {
                sh_flags[threadIdx.x] = 1;
            }
        }
        __syncthreads();
        SingleBlockBlellochScan(sh_flags, sh_flags_cumsum, blockDim.x);
        if (lb+threadIdx.x <= ub) {
            int elt = (in[lb+threadIdx.x] >> shift_right) & 3;
            if (bmask == elt) {
                sh_local_pfxsum[threadIdx.x] = 
                    threadIdx.x > 0 ? sh_flags_cumsum[threadIdx.x-1] : 0;
            }
        }
        __syncthreads();
    }
    dbg_out[threadIdx.x] = sh_local_pfxsum[threadIdx.x];
}

__global__ void CountBitPatterns(int* in, int N, int shift_right, int* local_block_sums, int* global_block_sums, int* global_bit_pattern_occs) {
    const int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    const int nthds = blockDim.x * gridDim.x;
    extern __shared__ char shmem[];
    short* sh_flags        = (short*)(&shmem[0]);
    short* sh_flags_cumsum = sh_flags + blockDim.x;
    const int num_blocks = (N-1) / blockDim.x + 1;

    for (int i=blockIdx.x * blockDim.x, bidx=blockIdx.x; i<N; i+=nthds, bidx+=gridDim.x) {
        int elt = (in[i+threadIdx.x] >> shift_right) & 3;
        for (int bmask=0; bmask<4; bmask++) {
            sh_flags[threadIdx.x] = 0;
            sh_flags_cumsum[threadIdx.x] = 0;
            __syncthreads();
            if (i + threadIdx.x < N) {
                if (elt == bmask) {
                    sh_flags[threadIdx.x] = 1;
                }
            }
            __syncthreads();
            SingleBlockBlellochScan(sh_flags, sh_flags_cumsum, blockDim.x);
            __syncthreads();
            if (i + threadIdx.x < N) {
                if (elt == bmask) {
                    local_block_sums[blockDim.x * bidx + threadIdx.x] = 
                        threadIdx.x > 0 ?
                            sh_flags_cumsum[threadIdx.x-1] : 0;
                }
            }
            if (threadIdx.x == 0) {
                int idx = num_blocks * bmask + bidx;
                //global_block_sums[num_blocks * bmask + 0] = 0;
                //if (idx < num_blocks * bmask + num_blocks - 1) 
                if(1)
                {
                    global_block_sums[idx] = sh_flags_cumsum[blockDim.x-1];
                } else {
                    global_bit_pattern_occs[bmask] = sh_flags_cumsum[blockDim.x-1];
                    printf("global bp %d=%d\n", bmask, sh_flags_cumsum[blockDim.x-1]);
                }
                //printf("[%d,%d] has %d %d's\n", i, i+blockDim.x-1, sh_flags_cumsum[blockDim.x-1], bmask);
            }
        }
    }
}

__global__ void ComputeBlockSums(int* in, int* out, int num_blocks) {
    SingleBlockBlellochScan(&(in[num_blocks  * blockIdx.x]),
                            &(out[num_blocks * blockIdx.x]),
                            num_blocks);
}

__global__ void Shuffle(int* in, int* out, int N, int shift_right, int* local_block_sums, int* global_block_sums) {
    const int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    const int nthds = blockDim.x * gridDim.x;
    const int num_blocks = (N-1) / blockDim.x + 1;

    int offsets123[4];
    offsets123[0] = 0;
    offsets123[1] = global_block_sums[num_blocks  -1];
    offsets123[2] = global_block_sums[num_blocks*2-1] + offsets123[1];
    offsets123[3] = global_block_sums[num_blocks*3-1] + offsets123[2];

    for (int i=blockIdx.x * blockDim.x, bidx=blockIdx.x; i<N; i+=nthds, bidx+=gridDim.x) {
        int elt = (in[i+threadIdx.x] >> shift_right) & 3;
        for (int bmask=0; bmask<4; bmask++) {
            if (i + threadIdx.x < N) {
                if (elt == bmask) {
                    const int lbs = local_block_sums[blockDim.x * bidx + threadIdx.x];
                    int goffset = (bidx > 0) ? global_block_sums[num_blocks * bmask + bidx - 1] : 0;
                    goffset += offsets123[bmask];
                    out[goffset + lbs] = in[i+threadIdx.x];
                }
            }
            __syncthreads();
        }
    }
}

void Test1() {
    const int N = 4;
    short h_in[N] = {0}, h_out[N] = {0};
    for (int i=0; i<N; i++) {
        h_in[i] = 1;
    }
    h_in[0] = 1;
    h_in[1] = 0;
    h_in[2] = 0;
    h_in[3] = 0;
    short *d_in, *d_out;
    musaMalloc(&d_in,  sizeof(short)*N);
    musaMalloc(&d_out, sizeof(short)*N);
    musaMemcpy(d_in,  h_in, sizeof(short)*N, musaMemcpyHostToDevice);
    SingleBlockBlellochScanTest<<<1, N, sizeof(short)*N*2>>>(d_in, d_out, N);
    musaMemcpy(h_out, d_out, sizeof(short)*N, musaMemcpyDeviceToHost);

    printf("h_in: ");
    for (int i=0; i<N; i++) {
        printf(" %2d", int(h_in[i]));
    }
    printf("\n");
    printf("h_out:");
    for (int i=0; i<N; i++) {
        printf(" %2d", int(h_out[i]));
    }
    printf("\n");
}

static const bool VERBOSE = false;

int main() {
    const int NT = 32;
    const int NB = 8;
    const int N = 256;
    int* h_dbg_out = new int[N], *d_dbg_out;
    musaMalloc(&d_dbg_out, sizeof(int)*N);
    int* h_in = new int[N], *d_in;
    musaMalloc(&d_in,  sizeof(int)*N);
    int* h_out = new int[N], *d_out;
    musaMalloc(&d_out, sizeof(int)*N);
    
    //int data[] = { 1,2,0,3, 0,1,1,0, 3,3,3,2, 1,2,2,0, 2,0,0,2, 1,2,3,2, 1,2,3,2, 2,3,2,1 };
    int data[N];
    for (int i=0; i<N; i++) {
        data[i] = rand();
    }
    memcpy(h_in, data, sizeof(data));

    const int num_blocks = (N-1) / NT + 1;
    int* d_scratch;
    musaMalloc(&d_scratch, sizeof(int)*(num_blocks*(4+NT+4) + 4));
    int* d_bit_patterns = d_scratch;
    int* d_local_block_sums = d_scratch + 4*num_blocks;
    int* d_global_block_sums = d_local_block_sums + NT*num_blocks;
    int* d_global_bit_pattern_occs = d_global_block_sums + num_blocks*4;

    int* h_bit_patterns = new int[4 * num_blocks];
    int* h_local_block_sums = new int[num_blocks * NT];
    int* h_global_block_sums   = new int[4 * num_blocks];

    musaMemcpy(d_in, h_in, sizeof(int)*N, musaMemcpyHostToDevice);

    //RadixSort<<<1, NT, sizeof(short)*N*3>>>(d_in, d_out, N, d_scratch, d_dbg_out);
    for (int shift_right = 0; shift_right < 32; shift_right += 2) {
        CountBitPatterns<<<NB, NT, sizeof(short)*NT*3>>>(d_in, N, shift_right, d_local_block_sums, d_bit_patterns, d_global_bit_pattern_occs);
        
        if (VERBOSE) {
            musaMemcpy(h_bit_patterns, d_bit_patterns, sizeof(int)*4*num_blocks, musaMemcpyDeviceToHost);
            printf("bit patterns: \n");
            for (int i=0; i<4; i++) {
                printf("%d :", i);
                for (int j=0; j<num_blocks; j++) {
                    printf(" %d", h_bit_patterns[num_blocks*i+j]);
                }
                printf("\n");
            }
            musaMemcpy(h_local_block_sums, d_local_block_sums, sizeof(int)*NT*num_blocks, musaMemcpyDeviceToHost);
            printf("local block sums:\n");
            for (int i=0; i<num_blocks; i++) {
                printf("Block %d:", i);
                for (int j=0; j<NT; j++) {
                    printf(" %d", h_local_block_sums[i*NT+j]);
                }
                printf("\n");
            }
        }

        ComputeBlockSums<<<4, 256>>>(d_bit_patterns, d_global_block_sums, num_blocks);

        if (VERBOSE) {
            musaMemcpy(h_global_block_sums, d_global_block_sums, sizeof(int)*4*num_blocks, musaMemcpyDeviceToHost);
            printf("global block sums: \n");
            for (int i=0; i<4; i++) {
                printf("%d :", i);
                for (int j=0; j<num_blocks; j++) {
                    printf(" %d", h_global_block_sums[num_blocks*i+j]);
                }
                printf("\n");
            }
        }

        Shuffle<<<1, NT>>>(d_in, d_out, N, shift_right, d_local_block_sums, d_global_block_sums);

        if (VERBOSE) {
            musaMemcpy(h_dbg_out, d_out, sizeof(int)*N, musaMemcpyDeviceToHost);
            printf("h_dbg_out:");
            for (int i=0; i<N; i++) { printf(" %d", h_dbg_out[i]); }
            printf("\n");
        }
        std::swap(d_in, d_out);
    }

    // Check result
    musaMemcpy(h_out, d_in, sizeof(int)*N, musaMemcpyDeviceToHost);
    for (int i=0; i<N-1; i++) {
        if (h_out[i] > h_out[i+1]) {
            printf("[%d]=%d, [%d]=%d\n", i, h_out[i], i+1, h_out[i+1]);
        }
    }
    printf("Passed!\n");

    return 0;
}