#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#define CHECK_CUDA(x)                                    \
  do {                                                   \
    AT_ASSERT(x.is_cuda(), #x " must be a CUDA tensor"); \
  } while (0)

#define CHECK_CONTIGUOUS(x)                                          \
  do {                                                               \
    AT_ASSERT(x.is_contiguous(), #x " must be a contiguous tensor"); \
  } while (0)

#define CHECK_IS_INT(x)                               \
  do {                                                \
    AT_ASSERT(x.scalar_type() == at::ScalarType::Int, \
              #x " must be an int tensor");           \
  } while (0)

#define CHECK_IS_FLOAT(x)                               \
  do {                                                  \
    AT_ASSERT(x.scalar_type() == at::ScalarType::Float, \
              #x " must be a float tensor");            \
  } while (0)


void ball_query_wrapper(int b, int n, int m, float radius,
                        int nsample, const float *new_xyz,
                        const float *xyz, int *idx);

at::Tensor ball_query(at::Tensor query, at::Tensor source, const float radius,
                      const int nsample) {
  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(source);
  CHECK_IS_FLOAT(query);
  CHECK_IS_FLOAT(source);

  if (query.is_cuda()) {
    CHECK_CUDA(source);
  }

  at::Tensor idx =
      torch::full({query.size(0), query.size(1), nsample}, -1,
                   at::device(query.device()).dtype(at::ScalarType::Int));

  if (query.is_cuda()) {
    ball_query_wrapper(source.size(0), source.size(1), query.size(1),
                       radius, nsample, query.data_ptr<float>(),
                       source.data_ptr<float>(), idx.data_ptr<int>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return idx;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ball_query", &ball_query);
}
