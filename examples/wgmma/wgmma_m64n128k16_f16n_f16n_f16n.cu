#include "cuda_fp16.h"
#include <cinttypes>
#include <iostream>

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
constexpr int szm = 64;
constexpr int szn = 128;
constexpr int szk = 16;

__host__ __device__ void printme(__half *ptr, int m, int n,
                                 int printingThread = 0) {
  int tid = 0;
#if defined(__CUDA_ARCH__)
  tid = threadIdx.x;
#endif
  if (tid == printingThread) {
    printf("===--- Thread[%d] - Tensor<%d, %d>  ---===\n", tid, m, n);

    printf("\t");
    for (int i = 0; i < n; ++i) {
      printf("%2d ", i);
    }
    printf("\n\t");
    for (int i = 0; i < szk; ++i) {
      printf("__ ");
    }
    printf("\n\n");

    for (int j = 0; j < m; ++j) {
      printf("[%2d]\t", j);
      for (int i = 0; i < n; ++i) {
        __half val = ptr[j * n + i];
        printf("%3.0f ", __half2float(val));
      }
      printf("\n");
    }
    printf("\n");
  }
}

__host__ __device__ void printme(float *ptr, int m, int n,
                                 int printingThread = 0) {
  int tid = 0;
#if defined(__CUDA_ARCH__)
  tid = threadIdx.x;
#endif
  if (tid == printingThread) {
    printf("===--- Thread[%d] - Tensor<%d, %d>  ---===\n", tid, m, n);

    printf("\t");
    for (int i = 0; i < n; ++i) {
      printf("%2d ", i);
    }
    printf("\n\t");
    for (int i = 0; i < szk; ++i) {
      printf("__ ");
    }
    printf("\n\n");

    for (int j = 0; j < m; ++j) {
      printf("[%2d]\t", j);
      for (int i = 0; i < n; ++i) {
        float val = ptr[j * n + i];
        printf("%3.0f ", val);
      }
      printf("\n");
    }
    printf("\n");
  }
}

enum class LayoutType : uint8_t {
  INTERLEAVE = 0,
  B128 = 1,
  B64 = 2,
  B32 = 3,
};

__device__ char const *to_string(LayoutType const &t) {
  switch (t) {
  case LayoutType::INTERLEAVE:
    return "INTERLEAVE";
  case LayoutType::B128:
    return "B128";
  case LayoutType::B64:
    return "B64";
  case LayoutType::B32:
    return "B32";
  }
  return nullptr;
}

union GmmaDescriptor {
  uint64_t desc_;
  uint32_t reg32_[2];
  uint16_t reg16_[4];

  // Bitfield implementation avoids the need for shifts in assignment
  struct {
    // start_address, bit [0,14), 4LSB not included
    uint16_t start_address_ : 14, : 2; // 14 bits [0,14), 2 bits unused
    // leading dimension byte offset, bit [16,30), 4LSB not included
    // For N: This is the stride from the first col to the second col of the 8x2
    // brick in INTERLEAVED
    //   Unused for all SWIZZLE_* layouts (and assumed to be 1)
    // For T: This is the stride from the first 8 rows to the next 8 rows.
    uint16_t leading_byte_offset_ : 14, : 2; // 14 bits [0,14), 2 bits unused
    // stride dimension byte offset, bit [32,46), 4LSB not included
    // For N: This is the stride from the first 8 rows to the next 8 rows.
    // For T: This is the stride fro mthe first 8 cols to the next 8 cols.
    uint16_t stride_byte_offset_ : 14, : 2; // 14 bits [0,14), 2 bits unused
    // base_offset, bit [49,52)
    // Valid only for SWIZZLE_128B and SWIZZLE_64B
    uint8_t : 1,
        base_offset_ : 3, : 4; // 1 bit unused, 3 bits [1,4), 4 bits unused
    // layout type, bit [62,64)
    // SWIZZLE_NONE = 0, SWIZZLE_32B = 3, SWIZZLE_64B = 2, SWIZZLE_128B = 1
    uint8_t : 6, layout_type_ : 2; // 6 bits unused, 2 bits [6,8)
  };

  // Decay to a uint64_t
  __device__ constexpr operator uint64_t() const noexcept { return desc_; }

  // Printer
  __device__ friend void print(GmmaDescriptor const &t) {
    printf("GmmaDescriptor: 0x%016" PRIx64 "\n", t.desc_);
    printf("  start_addr :  0x%04x\n", t.start_address_);
    printf("  leading_off:  0x%04x (%d)\n", t.leading_byte_offset_,
           t.leading_byte_offset_);
    printf("  stride_off :  0x%04x (%d)\n", t.stride_byte_offset_,
           t.stride_byte_offset_);
    printf("  base_offset:  0x%01x\n", t.base_offset_);
    printf("  layout_type:  0x%01x (%s)\n", t.layout_type_,
           to_string(static_cast<LayoutType>(t.layout_type_)));
  }
};

// https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma
fma(uint64_t const &desc_a, uint64_t const &desc_b, uint32_t &d00,
    uint32_t &d01, uint32_t &d02, uint32_t &d03, uint32_t &d04, uint32_t &d05,
    uint32_t &d06, uint32_t &d07, uint32_t &d08, uint32_t &d09, uint32_t &d10,
    uint32_t &d11, uint32_t &d12, uint32_t &d13, uint32_t &d14, uint32_t &d15,
    uint32_t &d16, uint32_t &d17, uint32_t &d18, uint32_t &d19, uint32_t &d20,
    uint32_t &d21, uint32_t &d22, uint32_t &d23, uint32_t &d24, uint32_t &d25,
    uint32_t &d26, uint32_t &d27, uint32_t &d28, uint32_t &d29, uint32_t &d30,
    uint32_t &d31) {

  // The operation of the form D = A*B is issued when the input predicate
  // argument scale-d is false.
  constexpr int scale_D = 0; // `p` in PTX

  // For the floating point variants of the wgmma.mma_async operation, each
  // element of the input matrices A and B can be negated by specifying the
  // value -1 for operands imm-scale-a and imm-scale-b respectively.  A value of
  // 1 can be used to avoid the negate operation. The valid values of
  // imm-scale-a and imm-scale-b  are -1 and 1.
  constexpr int scaleA = 1;
  constexpr int scaleB = 1;

  // Matrices A and B are stored in row-major and column-major format
  // respectively. For certain floating point variants, the input matrices A and
  // B can be transposed by specifying the value 1 for the immediate integer
  // arguments imm-trans-a and imm-trans-b respectively. A value of 0 can be
  // used to avoid the transpose operation. The valid values of imm-trans-a and
  // imm-trans-b are 0 and 1. The transpose operation is only supported for the
  // wgmma.mma_async variants with .f16/ .bf16 types on matrices accessed from
  // shared memory using matrix descriptors.
  constexpr int tnspA = 0;
  constexpr int tnspB = 1; // Switch to row-major
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900)
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f16.f16.f16 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
      " %8,  %9,  %10, %11, %12, %13, %14, %15, "
      " %16, %17, %18, %19, %20, %21, %22, %23, "
      " %24, %25, %26, %27, %28, %29, %30, %31},"
      " %32,"
      " %33,"
      " p,   %35, %36, %37, %38;\n"
      "}\n"
      : "+r"(d00), "+r"(d01), "+r"(d02), "+r"(d03), "+r"(d04), "+r"(d05),
        "+r"(d06), "+r"(d07), "+r"(d08), "+r"(d09), "+r"(d10), "+r"(d11),
        "+r"(d12), "+r"(d13), "+r"(d14), "+r"(d15), "+r"(d16), "+r"(d17),
        "+r"(d18), "+r"(d19), "+r"(d20), "+r"(d21), "+r"(d22), "+r"(d23),
        "+r"(d24), "+r"(d25), "+r"(d26), "+r"(d27), "+r"(d28), "+r"(d29),
        "+r"(d30), "+r"(d31)
      : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_D)), "n"(int32_t(scaleA)),
        "n"(int32_t(scaleB)), "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));
#endif
}

inline __device__ void fma(
    // The 64-bit register operands a-desc and b-desc are the matrix descriptors
    // which represent the multiplicand matrices A and B in shared memory
    // respectively.
    uint64_t const &desc_a, uint64_t const &desc_b,

    // Register operand d represents the accumulator matrix as well as the
    // destination matrix, distributed across the participating threads.
    float &d00, float &d01, float &d02, float &d03, float &d04, float &d05,
    float &d06, float &d07, float &d08, float &d09, float &d10, float &d11,
    float &d12, float &d13, float &d14, float &d15, float &d16, float &d17,
    float &d18, float &d19, float &d20, float &d21, float &d22, float &d23,
    float &d24, float &d25, float &d26, float &d27, float &d28, float &d29,
    float &d30, float &d31, float &d32, float &d33, float &d34, float &d35,
    float &d36, float &d37, float &d38, float &d39, float &d40, float &d41,
    float &d42, float &d43, float &d44, float &d45, float &d46, float &d47,
    float &d48, float &d49, float &d50, float &d51, float &d52, float &d53,
    float &d54, float &d55, float &d56, float &d57, float &d58, float &d59,
    float &d60, float &d61, float &d62, float &d63) {

  // The operation of the form D = A*B is issued when the input predicate
  // argument scale-d is false.
  constexpr int scale_D = 0; // `p` in PTX

  // For the floating point variants of the wgmma.mma_async operation, each
  // element of the input matrices A and B can be negated by specifying the
  // value -1 for operands imm-scale-a and imm-scale-b respectively.  A value of
  // 1 can be used to avoid the negate operation. The valid values of
  // imm-scale-a and imm-scale-b  are -1 and 1.
  constexpr int scaleA = 1;
  constexpr int scaleB = 1;

  // Matrices A and B are stored in row-major and column-major format
  // respectively. For certain floating point variants, the input matrices A and
  // B can be transposed by specifying the value 1 for the immediate integer
  // arguments imm-trans-a and imm-trans-b respectively. A value of 0 can be
  // used to avoid the transpose operation. The valid values of imm-trans-a and
  // imm-trans-b are 0 and 1. The transpose operation is only supported for the
  // wgmma.mma_async variants with .f16/ .bf16 types on matrices accessed from
  // shared memory using matrix descriptors.
  constexpr int tnspA = 0;
  constexpr int tnspB = 1; // Switch to row-major
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900)
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
      " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
      " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
      " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
      " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
      " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
      " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
      " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
      " %64,"
      " %65,"
      " p,    %67,  %68,  %69,  %70;\n"
      "}\n"
      : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03), "+f"(d04), "+f"(d05),
        "+f"(d06), "+f"(d07), "+f"(d08), "+f"(d09), "+f"(d10), "+f"(d11),
        "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17),
        "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23),
        "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29),
        "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35),
        "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41),
        "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47),
        "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53),
        "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59),
        "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63)
      : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_D)), "n"(int32_t(scaleA)),
        "n"(int32_t(scaleB)), "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));
#endif
}

inline __device__ void warpgroup_arrive() {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900)
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}

inline __device__ void warpgroup_commit_batch() {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900)
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}

template <int N> inline __device__ void warpgroup_wait() {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900)
  static_assert(N >= 0 && N <= 7,
                "_warpgroup.wait {N}; must be in range [0, 7]");
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
#endif
}

inline __device__ void warpgroup_fence_operand(float &reg) {
#if defined(__CUDA_ARCH__)
  asm volatile("" : "+f"(reg)::"memory");
#endif
}

__device__ uint32_t cast_smem_ptr_to_uint(void const *const ptr) {

  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

template <class T> inline __device__ void fillme(T *src, T *dst, int n, int m) {
  for (int i = threadIdx.x; i < n * m; i += blockDim.x)
    dst[i] = src[i];
}

__global__ void test(__half *lhs, __half *rhs, float *acc) {

  /////////////////////////////////////////////////////////////////////////////////
  // Prologue

  __shared__ float smema[4096]; //  64 x 16
  __shared__ float smemb[4096]; //  16 x 128

  __half *smem_a_ptr = reinterpret_cast<__half *>(smema);
  __half *smem_b_ptr = reinterpret_cast<__half *>(smemb);

  fillme(lhs, smem_a_ptr, szm, szk);
  fillme(rhs, smem_b_ptr, szk, szn);

  __syncthreads();

#ifdef DEBUG1
  printme(smem_a_ptr, szm, szk);
  printme(smem_b_ptr, szm, szk);
#endif

  /////////////////////////////////////////////////////////////////////////////////
  // GEMM

  // Create Matrix descriptors
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor

  // Matrix-LHS
  // start_addr :  0x0040
  // leading_off:  0x0001 (1)
  // stride_off :  0x0040 (64)
  // base_offset:  0x0
  // layout_type:  0x1 (B128)
  GmmaDescriptor desc_a;
  desc_a.layout_type_ = uint8_t(LayoutType::B128);

  uint32_t start_address = cast_smem_ptr_to_uint(smem_a_ptr);
  desc_a.start_address_ = start_address >> 4;

  constexpr uint8_t base_offset = 0;
  desc_a.base_offset_ = base_offset;

  desc_a.leading_byte_offset_ = uint32_t(1);
  desc_a.stride_byte_offset_ = uint32_t(64);

  // Matrix-RHS
  // GmmaDescriptor: 0x4000008000401c40
  // start_addr :  0x1c40
  // leading_off:  0x0040 (64)
  // stride_off :  0x0080 (128)
  // base_offset:  0x0
  // layout_type:  0x1 (B128)
  GmmaDescriptor desc_b;
  desc_b.layout_type_ = uint8_t(LayoutType::B128);

  start_address = cast_smem_ptr_to_uint(smem_b_ptr);
  desc_b.start_address_ = start_address >> 4;

  desc_b.base_offset_ = base_offset;

  desc_b.leading_byte_offset_ = uint32_t(64);
  desc_b.stride_byte_offset_ = uint32_t(128);

#ifdef DEBUG2
  if (threadIdx.x == 0) {
    printf("%p %p \n", smem_a_ptr, smem_b_ptr);
    print(desc_a);
    print(desc_b);
  }
#endif
  // Result registers
  float d[szm] = {0};
  float accum;
  warpgroup_fence_operand(accum);
  warpgroup_arrive();

  fma(desc_a, desc_b, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8],
      d[9], d[10], d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18],
      d[19], d[20], d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28],
      d[29], d[30], d[31], d[32], d[33], d[34], d[35], d[36], d[37], d[38],
      d[39], d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47], d[48],
      d[49], d[50], d[51], d[52], d[53], d[54], d[55], d[56], d[57], d[58],
      d[59], d[60], d[61], d[62], d[63]);

  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(accum);

#ifdef DEBUG3
  printme(d, 8, 8, 0);
  __syncthreads();
  printme(d, 8, 8, 1);
  __syncthreads();
  printme(d, 8, 8, 2);
  __syncthreads();
  printme(d, 8, 8, 3);
  __syncthreads();
  printme(d, 8, 8, 5);
  __syncthreads();
  printme(d, 8, 8, 6);
  __syncthreads();
  printme(d, 8, 8, 7);
  __syncthreads();
  printme(d, 8, 8, 8);
  __syncthreads();
  printme(d, 8, 8, 17);
  __syncthreads();
  printme(d, 8, 8, 31);
  __syncthreads();
  printme(d, 8, 8, 32);
  __syncthreads();
  printme(d, 8, 8, 64);
  __syncthreads();
  printme(d, 8, 8, 121);
  __syncthreads();
#endif

  /////////////////////////////////////////////////////////////////////////////////
  // Epilogue
  for (int i = 0; i < szm; ++i)
    acc[i * szn + threadIdx.x] = d[i];
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template <typename T> T *malloc_managed(size_t n, T init = T()) {
  T *data;
  bool failure =
      cudaMallocManaged((void **)&data, sizeof(T) * n) != cudaSuccess;
  if (failure) {
    printf("Setup failed\n");
    exit(-1);
  }
  return data;
}

template <typename T> T *malloc_managed() { return malloc_managed<T>(1); }

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

int main() {
  // Allocate memory
  __half *lhs = malloc_managed<__half>(szm * szk);
  __half *rhs = malloc_managed<__half>(szn * szk);
  float *acc = malloc_managed<float>(szn * szm);

  // Initialize data
  for (int i = 0; i < szm * szk; i++)
    lhs[i] = __float2half(i);
  for (int i = 0; i < szm * szn; i++)
    rhs[i] = __float2half(1);
  for (int i = 0; i < szn * szm; i++)
    acc[i] = 0.0f;

  // Luanch a kernel for warpgroup level GEMM
  test<<<1, 128>>>(lhs, rhs, acc);

#ifdef DEBUG4
  // print result
  printme<float>(acc, szm, szn);
#endif
  return 0;
}
