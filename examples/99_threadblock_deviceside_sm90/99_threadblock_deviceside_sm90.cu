#include <iostream>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "helper.h"

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Command line options parsing
struct Options {

  bool help;
  bool error, microkernel;

  int m, n, k, l, iterations;
  float alpha, beta;

  Options():
    help(false),
    error(false),
    microkernel(false),
    m(2048), n(2048), k(2048), l(1),
    alpha(1.f), beta(0.f),
    iterations(0)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }
    if (cmd.check_cmd_line_flag("microkernel")) 
      microkernel = true;
    cmd.get_cmd_line_argument("m", m, 2048);
    cmd.get_cmd_line_argument("n", n, 2048);
    cmd.get_cmd_line_argument("k", k, 2048);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "99_threadblock_deviceside_sm90\n\n"
      << "  This example showcases the use of CUTLASS's in IREE generated kernel\n"
      << "  performant kernels targeting NVIDIA's Hopper architecture.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   Sets the L extent (batch count) of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n";

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
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element>
bool initialize_block(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed=2023) {

  Element scope_max, scope_min;
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

  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, scope_max, scope_min, 0);

  return true;
}

////////////////////////////////////////////////////////////////////////////////
///          Templates for CUTLASS Threadblock-level matmul
////////////////////////////////////////////////////////////////////////////////

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementD = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

static constexpr int AlignmentA = 8;
static constexpr int AlignmentB = 8;
static constexpr int AlignmentC = 8;
static constexpr int AlignmentD = 8;
static constexpr int TILE_M = 128;
static constexpr int TILE_N = 128;
static constexpr int TILE_K = 64;

// using MainloopScheduleType = cutlass::gemm::KernelMultistage;
// using MainloopScheduleType = cutlass::gemm::KernelTma;
// using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecialized;
using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
// using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedCooperative;

// using EpilogueScheduleType = cutlass::epilogue::NoSmemWarpSpecialized;
using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecialized;
// using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecializedCooperative;

using StageCountType = _4;
// using StageCountType = cutlass::gemm::collective::StageCountAuto;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
  cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
  Shape<_128,_128,_64>, Shape<_1,_1,_1>,
  cutlass::epilogue::collective::EpilogueTileAuto,
  ElementD, ElementD,
  ElementC, LayoutC, AlignmentC,
  ElementC, LayoutD, AlignmentD,
  EpilogueScheduleType
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
  cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
  ElementA, LayoutA, AlignmentA,
  ElementB, LayoutB, AlignmentB,
  ElementD,
  Shape<_128,_128,_64>, Shape<_1,_1,_1>,
  std::conditional_t<std::is_same_v<StageCountType, cutlass::gemm::collective::StageCountAuto>,
      cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename CollectiveEpilogue::SharedStorage)>,
      StageCountType>,
  MainloopScheduleType
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
  Shape<int,int,int,int>,
  CollectiveMainloop,
  CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

using LayoutTagA = cutlass::gemm::detail::StrideToLayoutTagA_t<StrideA>;
using LayoutTagB = cutlass::gemm::detail::StrideToLayoutTagB_t<StrideB>;
using LayoutTagC = cutlass::gemm::detail::StrideToLayoutTagC_t<StrideC>;
using LayoutTagD = cutlass::gemm::detail::StrideToLayoutTagC_t<StrideD>;

using Params = typename GemmKernel::Params;

using MainloopParams = typename CollectiveMainloop::Params;
using EpilogueParams = typename CollectiveEpilogue::Params;
using TiledMma = typename CollectiveMainloop::TiledMma;
using TileShape = typename CollectiveMainloop::TileShape;

////////////////////////////////////////////////////////////////////////////////
///          CUTLASS Threadblock-level matmul
////////////////////////////////////////////////////////////////////////////////
template <class ElementA, class ElementB, class ElementC, bool hasLinalgFill,
          bool writeBack2Global>
inline __device__ void
ukernel_threadblock_gemm(ElementA *lhs, int64_t size_K, ElementB *rhs, int64_t size_N,
                         ElementC *res, int64_t size_M, ElementC *shmem,
                         ElementC fillValue, Params const& params) {
  // Dynamic shared memory base pointer
  extern __shared__ char shared_storage[];
  using namespace cute;
  using X = Underscore;
#if 0
  // Preconditions
  CUTE_STATIC_ASSERT(is_static<TileShape>::value);

  // Separate out problem shape for convenience
  // Optionally append _1s until problem shape is rank-4 in case its is only rank-3 (MNK)
  auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
  auto M = get<0>(problem_shape_MNKL);
  auto N = get<1>(problem_shape_MNKL);
  auto K = get<2>(problem_shape_MNKL);
  auto L = get<3>(problem_shape_MNKL);

  // Preconditions
  static_assert(rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
  static_assert(rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
  static_assert(rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
  static_assert(rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

  // Get the appropriate blocks for this thread block -- potential for thread block locality
  int thread_idx = int(threadIdx.x);
  auto blk_shape = TileShape{};                                                                // (BLK_M,BLK_N,BLK_K)
  auto [m_coord, n_coord, l_coord] = blockIdx;
  auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);                                        // (m,n,k,l)

  // Represent the full tensors
  Tensor mA_mkl = make_tensor(make_gmem_ptr(lhs), make_shape(M,K,L), params.mainloop.dA); //(m,k,l)
  Tensor mB_nkl = make_tensor(make_gmem_ptr(rhs), make_shape(N,K,L), params.mainloop.dB); //(n,k,l)

  // Get batch slice
  Tensor mA_mk = mA_mkl(_,_,l_coord);                                                                        // (m,k)
  Tensor mB_nk = mB_nkl(_,_,l_coord);                                                                        // (n,k)

  // Slice to get the tiles this thread block is responsible for
  Tensor gA_org = local_tile(mA_mk, blk_shape, take<0,3>(blk_coord_mnkl), Step<_1, X,_1>{});           // (BLK_M,BLK_K,k)
  Tensor gB_org = local_tile(mB_nk, blk_shape, take<0,3>(blk_coord_mnkl), Step< X,_1,_1>{});           // (BLK_N,BLK_K,k)
  
  Tensor gA = make_tensor(make_gmem_ptr(lhs), gA_org.layout());
  Tensor gB = make_tensor(make_gmem_ptr(rhs), gB_org.layout());

  // Compute tile residues for predication
  auto m_max_coord = M - size<0>(gA) * get<0>(blk_coord_mnkl);                             // M - BLK_M * m_coord
  auto n_max_coord = N - size<0>(gB) * get<1>(blk_coord_mnkl);                             // N - BLK_N * n_coord
  auto k_residue   = K - size<1>(gA) * size<2>(gA);                                        // K - BLK_K * k_coord_max
  auto residue_mnk = make_tuple(m_max_coord, n_max_coord, k_residue);

  // Allocate the tiled_mma and the accum for the (M,N) blk_shape
  TiledMma tiled_mma;
  Tensor accum = partition_fragment_C(tiled_mma, take<0,2>(blk_shape)); // (MMA,MMA_M,MMA_N)
  clear(accum);

  auto k_tile_iter  = cute::make_coord_iterator(shape<2>(gA));
  int  k_tile_count = size<2>(gA);
#if 0 
    // Perform the collective scoped MMA
    CollectiveMainloop collective_mma;
    collective_mma(
      accum,
      gA,
      gB,
      accum,
      k_tile_iter, k_tile_count,
      residue_mnk,
      thread_idx,
      shared_storage
    );
#else
    using SmemLayoutA = decltype(tile_to_shape(
      CollectiveMainloop::SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<CollectiveMainloop::DispatchPolicy::Stages>{})));
    using SmemLayoutB = decltype(tile_to_shape(
      CollectiveMainloop::SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<CollectiveMainloop::DispatchPolicy::Stages>{})));

    CollectiveMainloop::SharedStorage& storage = *reinterpret_cast<CollectiveMainloop::SharedStorage*>(shared_storage);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

    // Partition the copying of A and B tiles across the threads
    CollectiveMainloop::GmemTiledCopyA gmem_tiled_copy_a;
    CollectiveMainloop::GmemTiledCopyB gmem_tiled_copy_b;
    auto gmem_thr_copy_a = gmem_tiled_copy_a.get_slice(thread_idx);
    auto gmem_thr_copy_b = gmem_tiled_copy_b.get_slice(thread_idx);

    Tensor tAgA = gmem_thr_copy_a.partition_S(gA);                             // (ACPY,ACPY_M,ACPY_K,k)
    Tensor tAsA = gmem_thr_copy_a.partition_D(sA);                             // (ACPY,ACPY_M,ACPY_K,PIPE)
    Tensor tBgB = gmem_thr_copy_b.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = gmem_thr_copy_b.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

    // Tile MMA atom and compute thread partitions across A, B and C
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);

    // Allocate registers for pipelining
    Tensor tCsA = thr_mma.partition_A(sA);                                     // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thr_mma.partition_B(sB);                                     // (MMA,MMA_N,MMA_K,PIPE)

    Tensor tCrA = thr_mma.make_fragment_A(tCsA);                               // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);                               // (MMA,MMA_M,MMA_N,PIPE)

    //
    // Prologue
    //

    CUTLASS_PRAGMA_UNROLL
    for (int k_pipe = 0; k_pipe < CollectiveMainloop::DispatchPolicy::Stages-1; ++k_pipe) {
      copy(gmem_tiled_copy_a, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,k_pipe));
      copy(gmem_tiled_copy_b, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,k_pipe));
      cp_async_fence();
      ++k_tile_iter;
      --k_tile_count;
    }

    // Current pipe index in smem to read from
    int smem_pipe_read  = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = CollectiveMainloop::DispatchPolicy::Stages-1;

    //
    // Pipelined Main Loop
    //
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > -(CollectiveMainloop::DispatchPolicy::Stages-1); --k_tile_count)
    {
      // Copy gmem to smem before computing gemm on each k-pipe
      // pipe index in smem where the next gmem tile will be read into
      copy(gmem_tiled_copy_a, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,smem_pipe_write));
      copy(gmem_tiled_copy_b, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,smem_pipe_write));
      cp_async_fence();
      if (k_tile_count > 0) { ++k_tile_iter; }

      //
      // Compute on k_tile
      //
      warpgroup_fence_operand(accum);
      warpgroup_arrive();

      cp_async_wait<CollectiveMainloop::DispatchPolicy::Stages-2>();
      cute::gemm(tiled_mma, tCrA(_,_,_,smem_pipe_read), tCrB(_,_,_,smem_pipe_read), accum);
      warpgroup_commit_batch();

      //
      // Advance the pipe
      //
      ++smem_pipe_read;
      smem_pipe_read = (smem_pipe_read == CollectiveMainloop::DispatchPolicy::Stages) ? smem_pipe_read = 0 : smem_pipe_read;

      ++smem_pipe_write;
      smem_pipe_write = (smem_pipe_write == CollectiveMainloop::DispatchPolicy::Stages) ? smem_pipe_write = 0 : smem_pipe_write;

      // Wait for the pipeline MMAs to drain
      warpgroup_wait<0>();
      warpgroup_fence_operand(accum);
    }
#endif
    // Epilogue and write to gD
    CollectiveEpilogue epilogue{params.epilogue};
    epilogue(
      problem_shape_MNKL,
      blk_shape,
      blk_coord_mnkl,
      accum,
      tiled_mma,
      residue_mnk,
      thread_idx,
      shared_storage
    );
#endif 
}

////////////////////////////////////////////////////////////////////////////////
///          IREE Generated Kernel
////////////////////////////////////////////////////////////////////////////////  

__global__ void iree_generated_kernel(ElementA *lhs, ElementB *rhs, ElementC *res,
                                      int64_t M, int64_t N, int64_t K,
                                      CUTLASS_GRID_CONSTANT Params const params) {

  // IREE tiles outermost two loops of linalg.matmul with TILE_M, TILE_N
  for (int64_t tm = blockIdx.x * TILE_M; tm < M; tm += (gridDim.x * TILE_M)) {
    for (int64_t tn = blockIdx.y * TILE_N; tn < N; tn += (gridDim.y * TILE_N)) {
      ElementA *plhs = &lhs[tm * K];
      ElementB *prhs = &rhs[tn];
      ElementC *pres = &res[tm * N + tn];
      // The microkernel implements mainloop and epilogue using CUTLASS 
      ukernel_threadblock_gemm<ElementA, ElementB, ElementC, true, true>(
          plhs, K, prhs, N, res, M, nullptr, ElementC(0), params);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
struct ExampleRunner {
  //
  // Data members
  //

  /// Initialization
  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  uint64_t seed = 0;

  cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
  cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
  cutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
  cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D;
  cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_ref_D;

  //
  // Methods
  //

  bool verify(const ProblemShapeType& problem_size, float alpha, float beta) {
    auto [M, N, K, L] = problem_size;

    cutlass::TensorRef ref_A(block_A.get(), Gemm::LayoutA::packed({M, K}));
    cutlass::TensorRef ref_B(block_B.get(), Gemm::LayoutB::packed({K, N}));
    cutlass::TensorRef ref_C(block_C.get(), Gemm::LayoutC::packed({M, N}));
    cutlass::TensorRef ref_D(block_ref_D.get(), Gemm::LayoutD::packed({M, N}));

    cutlass::reference::device::GemmComplex(
          {M, N, K},
          typename Gemm::EpilogueOutputOp::ElementCompute(alpha),
          ref_A,
          cutlass::ComplexTransform::kNone,
          ref_B,
          cutlass::ComplexTransform::kNone,
          typename Gemm::EpilogueOutputOp::ElementCompute(beta),
          ref_C,
          ref_D,
          typename Gemm::EpilogueOutputOp::ElementAccumulator(0.f),
          L,     // batch_count
          M * K, // batch_stride_A
          K * N, // batch_stride_B
          M * N, // batch_stride_C
          M * N  // batch_stride_D
        );

    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Reference kernel failed. Last CUDA error: "
                << cudaGetErrorString(result) << std::endl;
      return false;
    }

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    bool passed = cutlass::reference::device::BlockCompareEqual(block_ref_D.get(), block_D.get(), block_D.size());

    return passed;
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(const ProblemShapeType& problem_size) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;
    
    auto shapeA = make_shape(M,K,L);
    auto shapeB = make_shape(N,K,L);
    auto shapeC = make_shape(M, N, L);
    stride_A = make_cute_packed_stride(StrideA{}, shapeA);
    stride_B = make_cute_packed_stride(StrideB{}, shapeB);
    stride_C = make_cute_packed_stride(StrideC{}, shapeC);
    stride_D = make_cute_packed_stride(StrideD{}, shapeC);
#if CUTLASS_DEBUG_TRACE_LEVEL > 0
    std::cout << "===--- initialize ---===" << std::endl;
    std::cout << "shapeA: " << shapeA  << "\tStride A: " << stride_A << std::endl;
    std::cout << "shapeB: " << shapeB  << "\tStride B: " << stride_B << std::endl;
    std::cout << "shapeC: " << shapeC  << "\tStride C: " << stride_C << std::endl;
    std::cout << "shapeC: " << shapeC  << "\tStride D: " << stride_D << std::endl;
#endif
    block_A.reset(M * K * L);
    block_B.reset(K * N * L);
    block_C.reset(M * N * L);
    block_D.reset(M * N * L);
    block_ref_D.reset(M * N * L);

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
    initialize_block(block_C, seed + 2021);
#if CUTLASS_DEBUG_TRACE_LEVEL > 1000
    auto blk_shape = TileShape{};
    std::cout << "blk_shape: " << blk_shape << std::endl;
    
    auto ptrA = block_A.get();
    auto ptrB = block_B.get();
    std::cout << "ptrA: " << ptrA << std::endl;
    std::cout << "ptrB: " << ptrB << std::endl;

    auto gmemPtrA = make_gmem_ptr(ptrA);
    auto gmemPtrB = make_gmem_ptr(ptrB);
    

    cutlass::TensorRef ref_A(block_A.get(), Gemm::LayoutA::packed({M, K}));
    cutlass::TensorRef ref_B(block_B.get(), Gemm::LayoutB::packed({K, N}));
    std::cout << "ref_A.data(): " << ref_A.data() << std::endl;
    std::cout << "ref_B.data(): " << ref_B.data() << std::endl;

    std::cout << "===---   ---===" <<  std::endl;
    for(int m_coord = 0; m_coord < 16; ++m_coord) {
      for(int n_coord = 0; n_coord < 16; ++n_coord) {
        // auto [m_coord, n_coord, l_coord] = blockIdx;
        int l_coord = 0;
        auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord); 
        // std::cout << "blk_coord_mnkl: " << blk_coord_mnkl << std::endl;

        // Represent full matrix
        Tensor mA_mkl = make_tensor(gmemPtrA, shapeA, stride_A);
        Tensor mB_nkl = make_tensor(gmemPtrB, shapeB, stride_B);
        Tensor mA_mk = mA_mkl(_,_,l_coord);
        Tensor mB_nk = mB_nkl(_,_,l_coord);
        
        // Slice to get the tiles this thread block is responsible for
        using X = Underscore;
        Tensor gA = local_tile(mA_mk, blk_shape, take<0,3>(blk_coord_mnkl), Step<_1, X,_1>{});
        Tensor gB = local_tile(mB_nk, blk_shape, take<0,3>(blk_coord_mnkl), Step< X,_1,_1>{});

        // Compute tile residues for predication
        auto m_max_coord = M - size<0>(gA) * get<0>(blk_coord_mnkl);
        auto n_max_coord = N - size<0>(gB) * get<1>(blk_coord_mnkl);
        auto k_residue   = K - size<1>(gA) * size<2>(gA);
        auto residue_mnk = make_tuple(m_max_coord, n_max_coord, k_residue);

        // std::cout << "+-- mA_mkl.layout(): " << mA_mkl.layout() 
        //           << "\tmA_mk.layout(): " << mA_mk.layout() 
        //           << "\tgA.layout(): " << gA.layout() 
        //           << std::endl;
        
        // std::cout << "+-- mB_nkl.layout(): " << mB_nkl.layout() 
        //           << "\tmB_nk.layout(): " << mB_nk.layout() 
        //           << "\tgB.layout(): " << gB.layout() 
        //           << std::endl;
        
        std::cout << "blk_coord_mnkl: " << blk_coord_mnkl
                  << "+-- residue_mnk: " << residue_mnk << std::endl;

        // Allocate the tiled_mma and the accumulators for the (M,N) blk_shape
        TiledMma tiled_mma;
        Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));
        // std::cout << "+-- accumulators.layout(): " << accumulators.layout() << std::endl;

        auto k_tile_iter  = cute::make_coord_iterator(shape<2>(gA));
        int  k_tile_count = size<2>(gA);
        // std::cout << "k_tile_iter: " << k_tile_count << std::endl;
        // std::cout << "k_tile_count: " << k_tile_count << std::endl;
      }
    }
    std::cout << "===---   ---===" <<  std::endl;
#endif
  }

  bool run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};

    initialize(problem_size);

    typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {block_A.get(), stride_A, block_B.get(), stride_B},
      {{options.alpha, options.beta}, block_C.get(), stride_C, block_D.get(), stride_D},
      hw_info
    };

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "This kernel is not supported. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return false;
    }

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to initialize the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return false;
    }
    // Paramters for microkernel 
    Params params = gemm_op.get_params();
    // MainloopParams mainloopParams = params.mainloop;
    // EpilogueParams epilogueParams = params.epilogue;
    
    dim3 const grid = gemm_op.get_grid_shape(params);
    dim3 constexpr block(size(TiledMma{}) ,1 , 1);
    int smem_size = GemmKernel::SharedStorageSize;
    
    #if CUTLASS_DEBUG_TRACE_LEVEL > 0
      print("Kernel Shape <<< (%d, %d, %d) , (%d, %d, %d), smem = %d >>> : ", grid.x, grid.y, grid.z, block.x, block.y, block.z, smem_size);
    #endif 

    ElementA *lhs = block_A.get();
    ElementB *rhs = block_B.get();
    ElementC *res = block_D.get();

    cudaError_t result;
    if(options.microkernel) {
      CUTLASS_TRACE_HOST("microkernel::run()");
      if (smem_size >= (48 << 10)) {
        result = cudaFuncSetAttribute(iree_generated_kernel,
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
            iree_generated_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

        if (result != cudaSuccess) {
          std::cerr << "cudaFuncSetAttribute/ "
                      "cudaFuncAttributePreferiree_kernel_"
                      "microkernelredSharedMemoryCarveout failed: "
                    << cudaGetErrorString(result) << std::endl;
          return 1;
        }
      }
      iree_generated_kernel<<<grid, block, smem_size>>>(lhs, rhs, res, options.m, options.n, options.k, params);
    } else {
      // Run the GEMM
      status = gemm_op.run();
    }
    result = cudaGetLastError();
    if (result != cudaSuccess) {
      CUTLASS_TRACE_HOST("Kernel launch is failed with error " << cudaGetErrorString(result));
      return 0;
    }
    result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Device sync is failed : "
                << cudaGetErrorString(result) << std::endl;
      return 0;
    }

    // Verify that the result is correct
    bool passed = verify(problem_size, options.alpha, options.beta);
    if (!passed) {
      std::cerr << "Reference check failed" << std::endl;
    }

    // Run profiling loop
    Result prof;
    if (passed && options.iterations > 0) {
      GpuTimer timer;
      timer.start();
      for (int iter = 0; iter < options.iterations; ++iter) {      
        if(options.microkernel) {
          iree_generated_kernel<<<grid, block, smem_size>>>(lhs, rhs, res, options.m, options.n, options.k, params);
        } else {
          gemm_op.run();      
        }
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

    return passed;
  }

};


///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to print a description of the example run and its result
void print_result(const std::string& description, bool passed) {
  std::cout << description << ": " << (passed ? "Passed" : "Failed") << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {
  //
  // Parse options
  //
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

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  
  ExampleRunner runner;
  bool passed = runner.run(options, hw_info);
  print_result("GEMM operation : ", passed);

  return 0;
}
