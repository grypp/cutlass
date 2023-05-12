
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"
#include <iostream>

#define TILE_M 64
#define TILE_N 128
#define TILE_K 64

////////////////////////////////////////////////////////////////////////////////
///          Typenames for CUTLASS Threadblock-level matmul.
///          Ideally these are nested in templates, but we
///          are not using templates here and just hardcoding
///          the types for simplicity of the example)
////////////////////////////////////////////////////////////////////////////////

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = cutlass::half_t;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

// A matrix configuration
int const Stages = 3;
using ThreadblockShape = cutlass::gemm::GemmShape<TILE_M, TILE_N, TILE_K>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, TILE_K>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
using DefaultMma = typename cutlass::gemm::threadblock::DefaultMma<
    ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
    ElementAccumulator, LayoutC, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80, ThreadblockShape, WarpShape, InstructionShape, Stages,
    cutlass::arch::OpMultiplyAdd>;

////////////////////////////////////////////////////////////////////////////////
///          Typenames for CUTLASS Threadblock-level matmul.
///          Ideally these are nested in templates, but we
///          are not using templates here and just hardcoding
///          the types for simplicity of the example)
////////////////////////////////////////////////////////////////////////////////

using namespace cute;
using ArchTag = cutlass::arch::Sm90;

using TileShape_MNK = Shape<_64, _128, _64>;
using ClusterShape_MNK = Shape<_1, _1, _1>;

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, cutlass::half_t,
        LayoutA, AlignmentA, cutlass::half_t, LayoutB, AlignmentB, float,
        TileShape_MNK, ClusterShape_MNK,
        cutlass::gemm::collective::StageCountAuto,
        cutlass::gemm::KernelTma>::CollectiveOp;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape_MNK,
        ClusterShape_MNK, cutlass::epilogue::collective::EpilogueTileAuto,
        float, float, cutlass::half_t, LayoutC, AlignmentC, cutlass::half_t,
        LayoutC, AlignmentC,
        cutlass::epilogue::NoSmemWarpSpecialized>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using Params = typename GemmKernel::Params;

using CollectiveMainLoop = typename Gemm::GemmKernel::CollectiveMainloop;
using MainloopParams = typename CollectiveMainLoop::Params;
using TileShape = typename CollectiveMainLoop::TileShape;
using TiledMma = typename CollectiveMainLoop::TiledMma;

using CollectiveEpilogue = Gemm::GemmKernel::CollectiveEpilogue;
using EpilogueParams = typename CollectiveEpilogue::Params;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

////////////////////////////////////////////////////////////////////////////////
///          CUTLASS Threadblock-level matmul
////////////////////////////////////////////////////////////////////////////////
template <class ElementA, class ElementB, class ElementC, bool hasLinalgFill,
          bool writeBack2Global>
__forceinline__ __device__ void
threadblock_gemm(ElementA *lhs, int64_t size_K, ElementB *rhs, int64_t size_N,
                 ElementC *res, int64_t size_M, ElementC *shmem,
                 ElementC fillValue, MainloopParams& mainloopParams,
                 EpilogueParams& epilogueParams) {
  // Dynamic shared memory base pointer
  extern __shared__ char shared_storage[];

  using namespace cute;
  using X = Underscore;

  // Any Tensor Op MMA Atom in the WGMMA ISA is arch conditional to sm90a.
  #if ! defined(__CUDA_ARCH_FEAT_SM90_ALL)
    if constexpr(size<0>(typename TiledMma::AtomShape_MNK{}) == 64) {
      printf("ERROR : Arch conditional MMA instruction used without targeting sm90a compute capability. Aborting.\n");
      return;
    }
  #endif

  // Preconditions
  static_assert(rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
  static_assert(rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
  static_assert(rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
  static_assert(rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

  int thread_idx = int(threadIdx.x);
  int warp_idx   = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
  int lane_predicate = cute::elect_one_sync();

  // Issue Tma Descriptor Prefetch from a single thread 
  if ((warp_idx == 0) && lane_predicate) {
    CollectiveMainloop::prefetch_tma_descriptors(mainloopParams);
  }
  // General Problem Shape
  auto problem_shape = make_shape(size_M, size_N, size_K, 1);
  // Separate out problem shape for convenience
  // Optionally append _1s until problem shape is rank-4 in case its is only rank-3 (MNK)
  auto problem_shape_MNKL = append<4>(problem_shape, Int<1>{});
  auto M = get<0>(problem_shape_MNKL);
  auto N = get<1>(problem_shape_MNKL);
  auto K = get<2>(problem_shape_MNKL);
  auto L = get<3>(problem_shape_MNKL);

  // TMA requires special handling of strides to deal with coord codomain mapping
  // Represent the full tensors -- get these from TMA
  Tensor mA_mkl = mainloopParams.tma_load_a.get_tma_tensor(make_shape(M,K,L));                            // (m,k,l)
  Tensor mB_nkl = mainloopParams.tma_load_b.get_tma_tensor(make_shape(N,K,L));                            // (n,k,l)

  // Get the appropriate blocks for this thread block -- potential for thread block locality
  auto blk_shape = TileShape{};                                                                // (BLK_M,BLK_N,BLK_K)
  auto blk_coord = make_coord(_,_,_);                                                   // (m,n,k) -- defer the slice

  // Make tiled views
  Tensor gA_mkl = local_tile(mA_mkl, blk_shape, blk_coord, Step<_1, X,_1>{});                  // (BLK_M,BLK_K,m,k,l)
  Tensor gB_nkl = local_tile(mB_nkl, blk_shape, blk_coord, Step< X,_1,_1>{});                  // (BLK_N,BLK_K,n,k,l)

  // Compute m_coord, n_coord, and l_coord with their post-tiled shapes
  auto m_coord = idx2crd(int(blockIdx.x), shape<2>(gA_mkl));
  auto n_coord = idx2crd(int(blockIdx.y), shape<2>(gB_nkl));
  auto l_coord = idx2crd(int(blockIdx.z), shape<4>(gB_nkl));
  auto output_tile_coord = make_coord(m_coord, n_coord, _, l_coord);

  // Slice with m_coord and n_coord
  Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                                                       // (BLK_M,BLK_K,k)
  Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);                                                       // (BLK_N,BLK_K,k)

  // Allocate the tiled_mma and the accumulators for the (M,N) blk_shape
  TiledMma tiled_mma;
  Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));                   // (MMA,MMA_M,MMA_N)

  auto k_tile_iter  = cute::make_coord_iterator(shape<2>(gA));
  auto k_tile_count = size<2>(gA);

  // Perform the collective scoped MMA
  CollectiveMainloop collective_mma;

  collective_mma(
    gA, mainloopParams.tma_load_a,
    gB, mainloopParams.tma_load_b,
    accumulators,
    k_tile_iter, k_tile_count,
    thread_idx,
    shared_storage,
    mainloopParams
  );

  constexpr int BLK_M_RANK = rank<0>(blk_shape);
  bool m_oob = int(blockIdx.x) >= size<2>(gA_mkl);
  auto m_max_coord = unwrap(cute::transform(make_seq<BLK_M_RANK>{}, [&](auto i) {
      return  m_oob ? 0 : get<i>(M) - get<0,i>(blk_shape) * get<i>(m_coord);
    }));

  constexpr int BLK_N_RANK = rank<1>(blk_shape);
  bool n_oob = int(blockIdx.y) >= size<2>(gB_nkl);
  auto n_max_coord = unwrap(cute::transform(make_seq<BLK_N_RANK>{}, [&](auto i) {
      return  n_oob ? 0 : get<i>(N) - get<1,i>(blk_shape) * get<i>(n_coord);
    }));
  auto residue_mnk = make_tuple(m_max_coord, n_max_coord, Int<0>{});

  // Epilogue and write to gD
  CollectiveEpilogue epilogue{epilogueParams};
  epilogue(
    problem_shape_MNKL,
    blk_shape,
    output_tile_coord,
    accumulators,
    tiled_mma,
    residue_mnk,
    thread_idx,
    shared_storage
  );
}

////////////////////////////////////////////////////////////////////////////////
///          Compiler Generated Kernel
////////////////////////////////////////////////////////////////////////////////
__global__ void generated_kernel(ElementA *lhs, ElementB *rhs, ElementC *res,
                                 int64_t M, int64_t N, int64_t K,
                                 MainloopParams& mainloopParams,
                                 EpilogueParams& epilogueParams) {
  /* Here outermost two loops are tiled with TILE_M, TILE_N. threadblock_gemm is
   * the microkernel that uses cutlass. */

  /* The kernel is where iree can fuse consumer or producer of matmul in this
   * loop. Communication between microkernel and fused operatiosn  are done via
   * shared memory ( if possible registers) */
  for (int64_t tm = blockIdx.x * TILE_M; tm < M; tm += (gridDim.x * TILE_M)) {
    for (int64_t tn = blockIdx.y * TILE_N; tn < N; tn += (gridDim.y * TILE_N)) {
      ElementA *plhs = &lhs[tm * K];
      ElementB *prhs = &rhs[tn];
      ElementC *pres = &res[tm * N + tn];
      threadblock_gemm<ElementA, ElementB, ElementC, true, true>(
          plhs, K, prhs, N, pres, M, nullptr, ElementC(0), mainloopParams,
          epilogueParams);
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
  int m, n, k, iterations;
  Options()
      : help(false), error(false), m(8192), n(8192), k(8192), iterations(100) {}

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
    cmd.get_cmd_line_argument("iterations", iterations);
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
        << "  --k=<int>                   Sets the K extent of the GEMM\n"
        << "  --iterations=<int>          Number of profiling iterations to "
           "perform.\n\n";

    return out;
  }

  double gflops(double runtime_s) const {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

/// Result structure
struct Result {
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(double avg_runtime_ms = 0, double gflops = 0,
         cutlass::Status status = cutlass::Status::kSuccess,
         cudaError_t error = cudaSuccess)
      : avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status),
        error(error), passed(false) {}
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
      // return 1;
    }

    // Carveout 100% shared memory for this kernel.
    result = cudaFuncSetAttribute(
        generated_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    if (result != cudaSuccess) {
      std::cerr << "cudaFuncSetAttribute/ "
                   "cudaFuncAttributePreferiree_kernel_"
                   "microkernelredSharedMemoryCarveout failed: "
                << cudaGetErrorString(result) << std::endl;
      // return 1;
    }
  }

  ElementA *lhs = matrix_A.device_data();
  ElementB *rhs = matrix_B.device_data();
  ElementC *res = matrix_C_computed.device_data();
  printf("Shmem %d\n", smem_size);

  // General Problem Shape
  auto problem_shape =
      make_shape(problem_size.m(), problem_size.n(), problem_size.k(), 1);

  // Separate out problem shape for convenience
  // Optionally append _1s until problem shape is rank-4 in case its is only
  // rank-3 (MNK)
  auto problem_shape_MNKL = append<4>(problem_shape, Int<1>{});
  auto M = get<0>(problem_shape_MNKL);
  auto N = get<1>(problem_shape_MNKL);
  auto K = get<2>(problem_shape_MNKL);
  auto L = get<3>(problem_shape_MNKL);

  StrideA stride_a =
      make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
  StrideB stride_b =
      make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
  StrideC stride_c =
      make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
  StrideD stride_d =
      make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  typename Gemm::Arguments arguments =
      typename Gemm::Arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                               problem_shape,
                               {lhs, stride_a, rhs, stride_b},
                               {{alpha, beta}, res, stride_c, res, stride_d},
                               hw_info};

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  std::cout << "workspace size : " << workspace_size << "\n";

  Gemm gemm;
  cutlass::Status status = gemm.can_implement(arguments);
  CUTLASS_CHECK(status);

  status = gemm.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  Params params = gemm.get_params();
  MainloopParams mainloopParams = params.mainloop;
  EpilogueParams epilogueParams = params.epilogue;

  generated_kernel<<<grid, block, smem_size>>>(
      lhs, rhs, res, problem_size.m(), problem_size.n(), problem_size.k(),
      mainloopParams, epilogueParams);

  // Run profiling loop
  Result prof;
  if (options.iterations > 0) {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      generated_kernel<<<grid, block, smem_size>>>(
          lhs, rhs, res, problem_size.m(), problem_size.n(), problem_size.k(),
          mainloopParams, epilogueParams);
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    prof.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    prof.gflops = options.gflops(prof.avg_runtime_ms / 1000.0);

    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x'
              << options.k << std::endl;
    std::cout << "  Avg runtime: " << prof.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << prof.gflops << std::endl;
  }
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
