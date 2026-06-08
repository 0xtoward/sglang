#include <sgl_kernel/tensor.h>   // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/type.cuh>   // For fp16_t, bf16_t, fp32_t
#include <sgl_kernel/utils.h>    // For RuntimeCheck, div_ceil
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE, PDL helpers
#include <sgl_kernel/runtime.cuh>  // For get_blocks_per_sm / get_sm_count

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

namespace {

// ----------------------------------------------------------------
// Snake1d activation, fused into a single element-wise kernel.
//   out[b,c,t] = in[b,c,t] + 1/(alpha[c] + 1e-9) * sin(alpha[c] * in[b,c,t])^2
// Input/output are [B, C, T] contiguous; alpha is [C] (per-channel).
// Math is done in fp32 internally (more accurate than the per-op bf16/fp16
// rounding of the eager reference; matches it to within bf16 round-off).
// Channel index for a flat element i is (i / T) % C, so alpha stays a single
// scalar per (b,c) row and the global loads of in/out are fully coalesced.
// Supports out == in (in-place).
// ----------------------------------------------------------------
template <typename T, bool kUsePDL>
__global__ void snake1d_kernel(T* __restrict__ out,
                               const T* __restrict__ in,
                               const T* __restrict__ alpha,  // [C]
                               uint32_t C,
                               uint32_t Tlen,
                               uint32_t n_total) {
  // If using PDL, wait for the primary kernel before any global load.
  device::PDLWaitPrimary<kUsePDL>();

  const uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_total; i += stride) {
    const uint32_t c = (i / Tlen) % C;
    const float a = static_cast<float>(alpha[c]);
    const float x = static_cast<float>(in[i]);
    const float inv = 1.0f / (a + 1e-9f);
    const float s = sinf(a * x);
    out[i] = static_cast<T>(x + inv * s * s);
  }

  // If using PDL, signal the secondary kernel after all threads finished.
  device::PDLTriggerSecondary<kUsePDL>();
}

// ----------------------------------------------------------------
// Launcher: validate tensors and launch a grid-stride kernel capped at
// (SM count x max blocks/SM) so a long sequence does not over-subscribe.
// ----------------------------------------------------------------
template <typename T, bool kUsePDL>
void snake1d(tvm::ffi::TensorView out,
             tvm::ffi::TensorView in,
             tvm::ffi::TensorView alpha) {
  using namespace host;

  SymbolicSize B{"B"};
  SymbolicSize C{"C"};
  SymbolicSize Tt{"T"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  // in / out: [B, C, T] contiguous, same dtype/device.
  TensorMatcher({B, C, Tt})  //
      .with_dtype<T>()
      .with_device(device_)
      .verify(out)
      .verify(in);
  // alpha: [C], pinned to the same C as in/out.
  TensorMatcher({C})  //
      .with_dtype<T>()
      .with_device(device_)
      .verify(alpha);

  const uint32_t b = static_cast<uint32_t>(B.unwrap());
  const uint32_t c = static_cast<uint32_t>(C.unwrap());
  const uint32_t tlen = static_cast<uint32_t>(Tt.unwrap());
  const uint32_t n = b * c * tlen;
  RuntimeCheck(n > 0, "snake1d: num_elements must be > 0");
  const DLDevice device = device_.unwrap();

  constexpr uint32_t kBlockSize = 256;
  static const uint32_t max_occ =
      runtime::get_blocks_per_sm(snake1d_kernel<T, kUsePDL>, kBlockSize);
  static const uint32_t num_sm = runtime::get_sm_count(device.device_id);
  const uint32_t grid =
      std::min<uint32_t>(div_ceil(n, kBlockSize), max_occ * num_sm);

  LaunchKernel(grid, kBlockSize, device).enable_pdl(kUsePDL)(
      snake1d_kernel<T, kUsePDL>,
      static_cast<T*>(out.data_ptr()),
      static_cast<const T*>(in.data_ptr()),
      static_cast<const T*>(alpha.data_ptr()),
      c,
      tlen,
      n);
}

}  // namespace
