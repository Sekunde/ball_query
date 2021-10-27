#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_CHECK_ERRORS()                                           \
  do {                                                                \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
      fprintf(stderr, "CUDA kernel failed : %s\n%s at L:%d in %s\n",  \
              cudaGetErrorString(err), __PRETTY_FUNCTION__, __LINE__, \
              __FILE__);                                              \
      exit(-1);                                                       \
    }                                                                 \
  } while (0)

#define TOTAL_THREADS 512
inline int opt_n_threads(int work_size) {
  const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);

  return max(min(1 << pow_2, TOTAL_THREADS), 1);
}


// input: query(b, m, 3) source(b, n, 3)
// output: idx(b, m, nsample)
__global__ void ball_query_kernel(int b, int n, int m, float radius,
                                  int nsample,
                                  const float *__restrict__ query,
                                  const float *__restrict__ source,
                                  int *__restrict__ idx) {
  int batch_index = blockIdx.x;
  source += batch_index * n * 3;
  query += batch_index * m * 3;
  idx += m * nsample * batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;

  float radius2 = radius * radius;
  for (int j = index; j < m; j += stride) {
    float query_x = query[j * 3 + 0];
    float query_y = query[j * 3 + 1];
    float query_z = query[j * 3 + 2];
    for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k) {
      float source_x = source[k * 3 + 0];
      float source_y = source[k * 3 + 1];
      float source_z = source[k * 3 + 2];
      float d2 = (query_x - source_x) * (query_x - source_x) + 
			     (query_y - source_y) * (query_y - source_y) + 
				 (query_z - source_z) * (query_z - source_z);
      if (d2 < radius2) {
        //if (cnt == 0) {
        //  for (int l = 0; l < nsample; ++l) {
        //    idx[j * nsample + l] = k;
        //  }
        //}
        idx[j * nsample + cnt] = k;
        ++cnt;
      }
    }
  }
}


void ball_query_wrapper(int b, int n, int m, float radius,
                        int nsample, const float *query,
                        const float *source, int *idx) {
  ball_query_kernel<<<b, opt_n_threads(m), 0>>>(
      b, n, m, radius, nsample, query, source, idx);

  CUDA_CHECK_ERRORS();
}
