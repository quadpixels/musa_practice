#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <utility>

#include <musa.h>
#include <musa_runtime_api.h>

// Fast 4-way parallel radix sorting on GPUs
//
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
//     ... 0: [1 2 0 1 2] ----> [0 1 2 0 1] ----> [0 1 3 3 4] -- +0 -->  [ 0  1  3  3  4]
//     ... 1: [1 2 0 1 0] ----> [0 1 2 0 1] ----> [0 1 3 3 4] -- +6 -->  [ 6  7  9  9 10]
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

__global__ void CountBitPatterns(int* in, int N, int shift_right, int* local_block_sums, int* global_block_sums) {
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
            __syncthreads();
            if (threadIdx.x == 0) {
                int idx = num_blocks * bmask + bidx;
                global_block_sums[idx] = sh_flags_cumsum[blockDim.x-1];
            }
            __syncthreads();
        }
    }
}

void CountBitPatterns_CPU(int* in, int N, int shift_right, int* local_block_sums, int NT) {
    const int num_blocks = (N-1) / NT + 1;
    memset(local_block_sums, 0x00, sizeof(int)*num_blocks*NT);
    for (int bidx=0, i=0; i<N; i+=NT, bidx++) {
        for (int bmask=0; bmask<4; bmask++) {
            int count = 0;
            for (int tidx=0; tidx<NT; tidx++) {
                int elt = (in[i+tidx] >> shift_right) & 3;
                if (elt == bmask) {
                    local_block_sums[bidx*NT + tidx] = count;
                    count++;
                }
            }
        }
    }
}

__global__ void ComputeBlockSums(int* in, int* out, int num_blocks) {
    SingleBlockBlellochScan(&(in[num_blocks  * blockIdx.x]),
                            &(out[num_blocks * blockIdx.x]),
                            num_blocks);
}

void ComputeBlockSums_CPU(int* in, int* out, int num_blocks) {
    for (int i=0; i<4; i++) {
        int count = 0;
        for (int j=0; j<num_blocks; j++) {
            count += in[num_blocks*i + j];
            out[num_blocks*i + j] = count;
        }
    }
}

__global__ void Shuffle(int* keys_in, int* keys_out, int* values_in, int* values_out, int N, int shift_right, int* local_block_sums, int* global_block_sums) {
    const int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    const int nthds = blockDim.x * gridDim.x;
    const int num_blocks = (N-1) / blockDim.x + 1;

    int offsets123[4];
    offsets123[0] = 0;
    offsets123[1] = global_block_sums[num_blocks  -1];
    offsets123[2] = global_block_sums[num_blocks*2-1] + offsets123[1];
    offsets123[3] = global_block_sums[num_blocks*3-1] + offsets123[2];

    for (int i=blockIdx.x * blockDim.x, bidx=blockIdx.x; i<N; i+=nthds, bidx+=gridDim.x) {
        int elt = (keys_in[i+threadIdx.x] >> shift_right) & 3;
        for (int bmask=0; bmask<4; bmask++) {
            if (i + threadIdx.x < N) {
                if (elt == bmask) {
                    const int lbs = local_block_sums[blockDim.x * bidx + threadIdx.x];
                    int goffset = (bidx > 0) ? global_block_sums[num_blocks * bmask + bidx - 1] : 0;
                    goffset += offsets123[bmask];
                    keys_out[goffset + lbs] = keys_in[i+threadIdx.x];
                    values_out[goffset+lbs] = values_in[i+threadIdx.x];
                }
            }
        }
    }
}

void Shuffle_CPU(int* keys_in, int* keys_out, int* values_in, int* values_out, int N, int shift_right, int* local_block_sums, int* global_block_sums, int NT) {
    const int num_blocks = (N-1) / NT + 1;
    int offsets123[4];
    offsets123[0] = 0;
    offsets123[1] = global_block_sums[num_blocks  -1];
    offsets123[2] = global_block_sums[num_blocks*2-1] + offsets123[1];
    offsets123[3] = global_block_sums[num_blocks*3-1] + offsets123[2];

    for (int bidx=0, i=bidx*NT; i<N; bidx++, i+=NT) {
        for (int bmask=0; bmask<4; bmask++) {
            for (int tidx=0; tidx<NT; tidx++) {
                int elt = (keys_in[i+tidx] >> shift_right) & 3;
                if (i+tidx < N) {
                    if (elt == bmask) {
                        const int lbs = local_block_sums[bidx * NT + tidx];
                        int goffset = (bidx > 0) ? global_block_sums[num_blocks * bmask + bidx - 1] : 0;
                        goffset += offsets123[bmask];
                        keys_out[goffset + lbs] = keys_in[i + tidx];
                        values_out[goffset+lbs] = values_in[i+tidx];
                    }
                }
            }
        }
    }
}

static const bool VERBOSE = false;
static const bool CHECK_RESULTS = true;

float DeltaMillis(
        const std::chrono::steady_clock::time_point& t1,
        const std::chrono::steady_clock::time_point& t0
) {
    return (std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) * 0.001f;
};

#define EXAMPLE_DATA

int main() {
#ifdef EXAMPLE_DATA
    const int NT = 4;
    const int NB = 1;
    const int N = 32;
#else
    const int NT = 256;
    const int NB = 128;
    const int N = 10000000;
#endif
    int* h_dbg_out = new int[N], *d_dbg_out;
    musaMalloc(&d_dbg_out, sizeof(int)*N);
    int* h_keys_in = new int[N], *d_keys_in;
    int* h_values_in = new int[N], *d_values_in;
    musaMalloc(&d_keys_in,  sizeof(int)*N);
    musaMalloc(&d_values_in, sizeof(int)*N);
    int* h_keys_out = new int[N], *d_keys_out;
    int* h_keys_out1 = new int[N];
    musaMalloc(&d_keys_out, sizeof(int)*N);
    int* h_values_out = new int[N], *d_values_out;
    int* h_values_out1 = new int[N];
    musaMalloc(&d_values_out, sizeof(int)*N);
    
#ifdef EXAMPLE_DATA
    int data[]   = { 1,2,0,3, 0,1,1,0, 3,3,3,2,   1,2,2,0,     2,0,0,2,     1,2,3,2,     1,2,3,2,     2,3,2,1     };
    int values[] = { 0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15, 16,17,18,19, 20,21,22,23, 24,25,26,27, 28,29,30,31 };
    memcpy(h_keys_in, data, sizeof(int)*N);
    memcpy(h_values_in, values, sizeof(int)*N);
#else
    for (int i=0; i<N; i++) {
        h_keys_in[i]   = rand();
        h_values_in[i] = i;
    }
#endif

    const int num_blocks = (N-1) / NT + 1;
    int* d_scratch;
    musaMalloc(&d_scratch, sizeof(int)*(num_blocks*(4+NT+4) + 4));
    int* d_bit_patterns = d_scratch;
    int* d_local_block_sums = d_scratch + 4*num_blocks;
    int* d_global_block_sums = d_local_block_sums + NT*num_blocks;

    int* h_bit_patterns = new int[4 * num_blocks];
    int* h_local_block_sums = new int[num_blocks * NT];
    int* h_global_block_sums   = new int[4 * num_blocks];

    int* h_local_block_sums1 = new int[num_blocks * NT];
    int* h_global_block_sums1 = new int[4 * num_blocks];

    musaMemcpy(d_keys_in, h_keys_in, sizeof(int)*N, musaMemcpyHostToDevice);
    musaMemcpy(d_values_in, h_values_in, sizeof(int)*N, musaMemcpyHostToDevice);
    auto t0 = std::chrono::steady_clock::now();

#ifdef EXAMPLE_DATA
    for (int shift_right = 0; shift_right <= 2; shift_right += 2)
#else
    for (int shift_right = 0; shift_right < 32; shift_right += 2)
#endif
    {
        CountBitPatterns<<<NB, NT, sizeof(short)*NT*3>>>(d_keys_in, N, shift_right, d_local_block_sums, d_bit_patterns);
        
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
            CountBitPatterns_CPU(h_keys_in, N, shift_right, h_local_block_sums1, NT);
            printf("local block sums:\n");
            for (int i=0; i<num_blocks; i++) {
                printf("Block %d:", i);
                for (int j=0; j<NT; j++) {
                    printf(" %d", h_local_block_sums[i*NT+j]);
                }
                printf(" vs ");
                for (int j=0; j<NT; j++) {
                    printf(" %d", h_local_block_sums1[i*NT+j]);
                }
                printf("\n");
            }
        }

        if (CHECK_RESULTS) {
            musaMemcpy(h_keys_in, d_keys_in, sizeof(int)*N, musaMemcpyDeviceToHost);
            CountBitPatterns_CPU(h_keys_in, N, shift_right, h_local_block_sums1, NT);
            musaMemcpy(h_local_block_sums, d_local_block_sums, sizeof(int)*NT*num_blocks, musaMemcpyDeviceToHost);
            for (int i=0; i<num_blocks; i++) {
                for (int j=0; j<NT && NT*i+j<N; j++) {
                    int x0 = h_local_block_sums[ i*NT+j];
                    int x1 = h_local_block_sums1[i*NT+j];
                    if (x0 != x1) {
                        printf("shift_right=%d, Local block sum at block %d thd %d not matching: %d vs %d\n",
                            shift_right, i, j, x0, x1);
                        abort();
                    }
                }
            }

            musaMemcpy(h_bit_patterns, d_bit_patterns, sizeof(int)*4*num_blocks, musaMemcpyDeviceToHost);  // for ComputeBlockSums_CPU's usage before ComputeBlockSums overwrites d_bit_patterns
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

        if (CHECK_RESULTS) {
            musaMemcpy(h_global_block_sums, d_global_block_sums, sizeof(int)*4*num_blocks, musaMemcpyDeviceToHost);
            ComputeBlockSums_CPU(h_bit_patterns, h_global_block_sums1, num_blocks);
            for (int i=0; i<4; i++) {
                for (int j=0; j<num_blocks; j++) {
                    int x0 = h_global_block_sums[i*num_blocks + j];
                    int x1 = h_global_block_sums1[i*num_blocks + j];
                    if (x0 != x1) {
                        printf("shift_right=%d, Global block sum with pattern=%d, thd %d not matching: %d vs %d\n",
                            shift_right, i, j, x0, x1);
                        abort();
                    }
                }
            }
        }

        Shuffle<<<NB, NT>>>(d_keys_in, d_keys_out, d_values_in, d_values_out, N, shift_right, d_local_block_sums, d_global_block_sums);

        if (VERBOSE) {
            musaMemcpy(h_dbg_out, d_keys_out, sizeof(int)*N, musaMemcpyDeviceToHost);
            printf("h_dbg_out:");
            for (int i=0; i<N; i++) { printf(" %d", h_dbg_out[i]); }
            printf("\n");
        }

        if (CHECK_RESULTS) {
            musaDeviceSynchronize();
            Shuffle_CPU(h_keys_in, h_keys_out1, h_values_in, h_values_out1, N, shift_right, h_local_block_sums1, h_global_block_sums1, NT);
            musaMemcpy(h_keys_out, d_keys_out, sizeof(int)*N, musaMemcpyDeviceToHost);
            musaMemcpy(h_values_out, d_values_out, sizeof(int)*N, musaMemcpyDeviceToHost);
            for (int i=0; i<N; i++) {
                if (h_keys_out[i] != h_keys_out1[i]) {
                    printf("shift_right=%d, Shuffle, keys [%d] not matched: %d vs %d\n",
                        shift_right, i, h_keys_out[i], h_keys_out1[i]);
                }
                if (h_values_out[i] != h_values_out1[i]) {
                    printf("shift_right=%d, Shuffle, value [%d] not matched: %d vs %d\n",
                        shift_right, i, h_values_out[i], h_values_out1[i]);
                }
            }
            memcpy(h_values_in, h_values_out1, sizeof(int)*N);
        }

        std::swap(d_keys_in,   d_keys_out);
        std::swap(d_values_in, d_values_out);
    }

    musaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    float msecs = DeltaMillis(t1, t0);
    printf("Sorting %d elts: %g ms elapsed, %g M elts/s\n", N, msecs, N/(msecs/1000.0f)/1000000.0f);

    // Check result
    bool ok = true;
    musaMemcpy(h_keys_out, d_keys_in, sizeof(int)*N, musaMemcpyDeviceToHost);
    for (int i=0; i<N-1; i++) {
        if (h_keys_out[i] > h_keys_out[i+1]) {
            printf("[%d]=%d, [%d]=%d\n", i, h_keys_out[i], i+1, h_keys_out[i+1]);
            ok = false;
        }
    }
    if (ok) {
        printf("Passed!\n");
    } else {
        printf("Failed!\n");
        abort();
    }

    return 0;
}