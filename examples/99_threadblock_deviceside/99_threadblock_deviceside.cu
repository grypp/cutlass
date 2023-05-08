#include <cstdint>
#include <cstdio>

#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/core_io.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/tensor_view_io.h"

#define TILE_M 128
#define TILE_N 128
#define TILE_K 32

////////////////////////////////////////////////////////////////////////////////
///          Typenames for CUTLASS Threadblock-level matmul.
///          Ideally these are nested in templates, but we
///          are not using templates here and just hardcoding
///          the types for simplicity of the example)
////////////////////////////////////////////////////////////////////////////////

using ElementA = cutlass::tfloat32_t;
using LayoutA = cutlass::layout::RowMajor;
using ElementB = cutlass::tfloat32_t;
using LayoutB = cutlass::layout::RowMajor;
using ElementC = float;
using LayoutC = cutlass::layout::RowMajor;
using ElementAccumulator = float;

using ThreadblockShape = cutlass::gemm::GemmShape<TILE_M, TILE_N, TILE_K>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, TILE_K>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
int const kAlignmentA = 4;
int const kAlignmentB = 4;
int const Stages = 3;

using DefaultMma = typename cutlass::gemm::threadblock::DefaultMma<
    ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
    ElementAccumulator, LayoutC, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80, ThreadblockShape, WarpShape, InstructionShape, Stages,
    cutlass::arch::OpMultiplyAdd>;

////////////////////////////////////////////////////////////////////////////////
///          CUTLASS Threadblock-level matmul
////////////////////////////////////////////////////////////////////////////////
template <class ElementA, class ElementB, class ElementC, bool hasLinalgFill,
          bool writeBack2Global>
__forceinline__ __device__ void
threadblock_gemm(ElementA *lhs, int64_t size_K, ElementB *rhs, int64_t size_N,
                 ElementC *res, int64_t size_M, ElementC *shmem,
                 ElementC fillValue) {
  // CUTLASS Threadblock-level multistage matrix multiply-accumulate
  // pipeline
  using ThreadblockMma = typename DefaultMma::ThreadblockMma;
  using IteratorA = typename ThreadblockMma::IteratorA;
  using IteratorB = typename ThreadblockMma::IteratorB;

  const int SZ_K = size_K;
  const int SZ_N = size_N;
  const int SZ_M = size_M;

  // Set entire matrix as the problem size
  cutlass::gemm::GemmCoord problem_size(SZ_M, SZ_N, SZ_K);

  cutlass::gemm::GemmCoord tb_tile_offset = {0, 0, 0};

  // Dynamic shared memory base pointer
  extern __shared__ ElementC GemmSharedStorageBase[];

  // Declare pointer to dynamic shared memory.
  typename ThreadblockMma::SharedStorage *shared_storage =
      reinterpret_cast<typename ThreadblockMma::SharedStorage *>(
          GemmSharedStorageBase);

  cutlass::MatrixCoord tb_offset_A{
      tb_tile_offset.m() * ThreadblockMma::Shape::kM, tb_tile_offset.k()};

  cutlass::MatrixCoord tb_offset_B{
      tb_tile_offset.k(), tb_tile_offset.n() * ThreadblockMma::Shape::kN};

  // Compute position within threadblock
  int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
  int warp_id = __shfl_sync(0xffffffff, threadIdx.y, 0);
  int lane_id = tb_thread_id & 0x1f;

  typename IteratorA::Params params_A(
      cutlass::layout::RowMajor::packed({problem_size.m(), problem_size.k()}));

  typename IteratorB::Params params_B(
      cutlass::layout::RowMajor::packed({problem_size.k(), problem_size.n()}));

  // Construct iterators to A and B operands
  typename ThreadblockMma::IteratorA iterator_A(
      params_A, lhs, {problem_size.m(), problem_size.k()}, tb_thread_id,
      tb_offset_A);

  typename ThreadblockMma::IteratorB iterator_B(
      params_B, rhs, {problem_size.k(), problem_size.n()}, tb_thread_id,
      tb_offset_B);

  typename ThreadblockMma::Operator::IteratorC iterator_LoadC(
      {res, problem_size.n()}, threadIdx.x);

  // Construct thread-scoped matrix multiply
  ThreadblockMma mma(*shared_storage, tb_thread_id, warp_id, lane_id);

  typename ThreadblockMma::FragmentC accumDest;
  accumDest.clear();

  int gemm_k_iterations = (problem_size.k() + ThreadblockMma::Shape::kK - 1) /
                          ThreadblockMma::Shape::kK;

  if (!hasLinalgFill) {
    typename ThreadblockMma::FragmentC accumSrc;

    // Set the offset
    iterator_LoadC.add_tile_offset(
        {(tb_tile_offset.m() * ThreadblockMma::WarpCount::kM) +
             (warp_id % ThreadblockMma::WarpCount::kM),
         (tb_tile_offset.n() * ThreadblockMma::WarpCount::kN) +
             (warp_id / ThreadblockMma::WarpCount::kM)});

    // Clear the fragment
    accumSrc.clear();

    // Load C as source accumulator
    iterator_LoadC.load(accumSrc);

    // Compute threadblock-scoped matrix multiply-add
    mma(gemm_k_iterations, accumDest, iterator_A, iterator_B, accumSrc);
  } else {
    // Compute threadblock-scoped matrix multiply-add
    mma(gemm_k_iterations, accumDest, iterator_A, iterator_B, accumDest);
  }

  int const kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementC>::value;

  using EpilogueOutputOp =
      cutlass::epilogue::thread::LinearCombination<ElementC, kElementsPerAccess,
                                                   ElementC, ElementC>;
  using EpilogueThreadBlock =
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
          ThreadblockShape, typename ThreadblockMma::Operator,
          ThreadblockMma::Policy::kPartitionsK, EpilogueOutputOp,
          EpilogueOutputOp::kCount>::Epilogue;

  if (!writeBack2Global) {
    int total_elements = accumDest.size();
    ElementC *offset_shmem =
        &GemmSharedStorageBase[tb_thread_id * total_elements];
    for (int i = 0; i < accumDest.size(); ++i) {
      offset_shmem[i] = accumDest[i];
      if (hasLinalgFill)
        offset_shmem[i] += fillValue;
    }
    __syncthreads();
  } else {
    // assume identity swizzle
    cutlass::MatrixCoord threadblock_offset{
        tb_tile_offset.m() * ThreadblockMma::Shape::kM,
        tb_tile_offset.n() * ThreadblockMma::Shape::kN};

    // Create Layout
    typename EpilogueThreadBlock::OutputTileIterator::Params params_D(
        cutlass::layout::RowMajor::packed(
            {problem_size.m(), problem_size.n()}));

    // Tile iterator loading from source tensor.
    typename EpilogueThreadBlock::OutputTileIterator iterator_D(
        params_D, res, problem_size.mn(), tb_thread_id, threadblock_offset);

    typename EpilogueOutputOp::Params params_output_op;

    EpilogueOutputOp output_op(params_output_op);

    // Reuse the same shared memory that we used for the inputs
    typename EpilogueThreadBlock::SharedStorage *e_shared_storage =
        reinterpret_cast<typename EpilogueThreadBlock::SharedStorage *>(
            GemmSharedStorageBase);

    EpilogueThreadBlock epilogue(*e_shared_storage, tb_thread_id, warp_id,
                                 lane_id);

    // Execute the epilogue operator to update the destination tensor.
    if (hasLinalgFill) {
      epilogue(output_op, iterator_D, accumDest);
    } else {
      typename EpilogueThreadBlock::OutputTileIterator::Params params_C(
          cutlass::layout::RowMajor::packed(
              {problem_size.m(), problem_size.n()}));
      // Use matrix C as a source
      typename EpilogueThreadBlock::OutputTileIterator iterator_C(
          params_C, res, problem_size.mn(), tb_thread_id, threadblock_offset);
      // Use matrix C as a source
      epilogue(output_op, iterator_D, accumDest, iterator_C);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
///          Generated Kernel
////////////////////////////////////////////////////////////////////////////////
__global__ void generated_kernel(ElementA *lhs, ElementB *rhs, ElementC *res,
                                 int64_t M, int64_t N, int64_t K) {
  for (int64_t tm = blockIdx.x * TILE_M; tm < M; tm += (gridDim.x * TILE_M)) {
    for (int64_t tn = blockIdx.y * TILE_N; tn < N; tn += (gridDim.y * TILE_N)) {
      ElementA *plhs = &lhs[tm * K];
      ElementB *prhs = &rhs[tn];
      ElementC *pres = &res[tm * N + tn];
      threadblock_gemm<ElementA, ElementB, ElementC, true, true>(
          plhs, K, prhs, N, pres, M, 0, 0);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
///          Host Part
////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a tensor view
template <typename Element, typename Layout>
bool initialize_tensor(cutlass::TensorView<Element, Layout> view,
                       cutlass::Distribution::Kind dist_kind, uint64_t seed) {
  if (dist_kind == cutlass::Distribution::Uniform) {
    double scope_max, scope_min;
    int bits_input = cutlass::sizeof_bits<Element>::value;

    if (bits_input == 1) {
      scope_max = 2;
      scope_min = 0;
    } else if (bits_input <= 8) {
      scope_max = 2;
      scope_min = -2;
    } else {
      scope_max = 8;
      scope_min = -8;
    }

    cutlass::reference::host::TensorFillRandomUniform(view, seed, scope_max,
                                                      scope_min, 0);
  } else if (dist_kind == cutlass::Distribution::Identity) {
    cutlass::reference::host::TensorFillIdentity(view);
  } else if (dist_kind == cutlass::Distribution::Gaussian) {
    cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
  } else if (dist_kind == cutlass::Distribution::Sequential) {
    cutlass::reference::host::BlockFillSequential(view.data(), view.capacity());
  } else {
    // TODO: Implement the rest
    std::cerr << "Not implemented";
    return false;
  }

  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Command line options parsing
struct Options {

  bool help;
  bool error;
  int m, n, k;
  Options() : help(false), error(false), m(8192), n(8192), k(8192) {}

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m, 8192);
    cmd.get_cmd_line_argument("n", n, 8192);
    cmd.get_cmd_line_argument("k", k, 8192);
  }

  /// Prints the usage statement.
  std::ostream &print_usage(std::ostream &out) const {

    out << "49_hopper_with_collective_builder\n\n"
        << "  This example showcases the use of CUTLASS's collective operation "
           "builders to easily construct\n"
        << "  performant kernels targeting NVIDIA's Hopper architecture.\n\n"
        << "Options:\n\n"
        << "  --help                      If specified, displays this usage "
           "statement\n\n"
        << "  --m=<int>                   Sets the M extent of the GEMM\n"
        << "  --n=<int>                   Sets the N extent of the GEMM\n"
        << "  --k=<int>                   Sets the K extent of the GEMM\n";

    return out;
  }
};

int main(int argc, char const **args) {
  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  dim3 grid(options.m / TILE_M, options.n / TILE_N);
  dim3 block(32, TILE_N / 32, 1);

  // CUTLASS Threadblock-level multistage matrix multiply-accumulate pipeline
  using ThreadblockMma = typename DefaultMma::ThreadblockMma;
  // Create a GEMM
  cutlass::gemm::GemmCoord problem_size(options.m, options.n, options.k);

  float alpha = 1.0f;
  float beta = 0.0f;

  cutlass::HostTensor<ElementA, LayoutA> matrix_A;
  cutlass::HostTensor<ElementB, LayoutB> matrix_B;
  cutlass::HostTensor<ElementC, LayoutC> matrix_C_computed;
  cutlass::HostTensor<ElementC, LayoutC> matrix_C_reference;

  // Allocate device and host memory
  matrix_A.resize(problem_size.mk());
  matrix_B.resize(problem_size.kn());
  matrix_C_computed.resize(problem_size.mn());
  matrix_C_reference.resize(problem_size.mn(), false);

  //
  // initialize device memory
  //
  uint64_t seed = 2080;
  cutlass::Distribution::Kind init_A = cutlass::Distribution::Uniform;
  cutlass::Distribution::Kind init_B = cutlass::Distribution::Uniform;
  cutlass::Distribution::Kind init_C = cutlass::Distribution::Uniform;

  initialize_tensor(matrix_A.host_view(), init_A, seed + 2019);
  initialize_tensor(matrix_B.host_view(), init_B, seed + 2018);
  initialize_tensor(matrix_C_computed.host_view(), init_C, seed + 2017);
  cutlass::reference::host::TensorCopy(matrix_C_reference.host_view(),
                                       matrix_C_computed.host_view());

  // Sync device memory (copy host to device)
  matrix_A.sync_device();
  matrix_B.sync_device();
  matrix_C_computed.sync_device();

  cudaError_t result;

  // If requires more than 48KB: configure for extended, dynamic shared memory
  int smem_size = int(sizeof(typename ThreadblockMma::SharedStorage));
  std::cout << "smem_size: " << smem_size << std::endl;
  cudaDeviceProp properties;
  int device_idx;

  result = cudaGetDevice(&device_idx);
  if (result != cudaSuccess) {
    throw std::runtime_error("cudaGetDevice() API call failed.");
  }

  result = cudaGetDeviceProperties(&properties, device_idx);
  if (result != cudaSuccess) {
    throw std::runtime_error("cudaGetDeviceProperties() failed");
  }

  if (properties.sharedMemPerBlockOptin < smem_size) {
    std::cerr << "Shared memory size (" << properties.sharedMemPerBlockOptin
              << " bytes) "
              << "exceeds the device limit. Please use a device with more "
                 "shared memory."
              << std::endl;
    throw std::runtime_error("cudaGetDeviceProperties() failed");
  }

  if (smem_size >= (48 << 10)) {
    result = cudaFuncSetAttribute(generated_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  smem_size);
    if (result != cudaSuccess) {
      std::cerr << "cudaFuncSetAttribute / "
                   "cudaFuncAttributeMaxDynamicSharedMemorySize failed: "
                << cudaGetErrorString(result) << std::endl;
      return 1;
    }

    // Carveout 100% shared memory for this kernel.
    result = cudaFuncSetAttribute(
        generated_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    if (result != cudaSuccess) {
      std::cerr << "cudaFuncSetAttribute/ "
                   "cudaFuncAttributePreferiree_kernel_"
                   "microkernelredSharedMemoryCarveout failed: "
                << cudaGetErrorString(result) << std::endl;
      return 1;
    }
  }

  ElementA *lhs = matrix_A.device_data();
  ElementB *rhs = matrix_B.device_data();
  ElementC *res = matrix_C_computed.device_data();
  printf("Shmem %d\n", smem_size);

  generated_kernel<<<grid, block, smem_size>>>(
      lhs, rhs, res, problem_size.m(), problem_size.n(), problem_size.k());

  //
  // Check error code
  //
  result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cerr << " kernel error: " << cudaGetErrorString(result);
  }

  matrix_C_computed.sync_host();

  // VERFIY HERE
  cutlass::reference::host::Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC,
                                 LayoutC, ElementC, ElementC>
      reference_gemm;
  reference_gemm(problem_size, ElementC(alpha), matrix_A.host_view(),
                 matrix_B.host_view(), ElementC(beta),
                 matrix_C_reference.host_view());

  bool passed = cutlass::reference::host::TensorEquals(
      matrix_C_computed.host_view(), matrix_C_reference.host_view());

  if (!passed) {
    std::cout << __FILE__ << ":" << __LINE__ << "  "
              << "A:\n"
              << matrix_A.host_view() << "\n"
              << "B:\n"
              << matrix_B.host_view() << "\n"
              << "Reference:\n"
              << matrix_C_reference.host_view() << "\n"
              << "Computed:\n"
              << matrix_C_computed.host_view() << "\n";
  }

  printf("VERIFY: %s\n", passed ? "SUCCESS" : "FAIL");

  return 0;
}
