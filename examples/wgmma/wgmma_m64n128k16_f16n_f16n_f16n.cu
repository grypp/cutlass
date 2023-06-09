// Compile with
// nvcc -w -arch=sm_90a wgmma_m64n128k16_f16n_f16n_f16n.cu -DDEBUG1 -DDEBUG2 -DDEBUG3 -DDEBUG4 -DDEBUG5 


#include <cinttypes>
#include <iostream>

#include "cuda_fp16.h"
#include "cuda_runtime.h"

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
constexpr int sz64 = 64;
constexpr int sz128 = 128;
constexpr int sz16 = 16;
constexpr int core8 = 8;

__device__ void printme1(float *ptr, int m, int printingThread = 0) {
  int tid = 0;
#if defined(__CUDA_ARCH__)
  tid = threadIdx.x;
  char * target = "GPU";
#else
  char * target = "CPU";
#endif 
  if (tid == printingThread) {
    printf("==-- %s Thread[%3d] - Tensor<%d> : ", target, tid, m);
    for (int i = 0; i < m; i++) {
      printf("%3.0f ", ptr[i]);
    }
    printf("\n");
  }
}

__device__ void printme1(__half *ptr, int m, int printingThread = 0) {
#if defined(__CUDA_ARCH__)
  int tid = threadIdx.x;
  char * target = "GPU";
#else
int tid = 0;
  char * target = "CPU";
#endif 
  if (tid == printingThread) {
    printf("==-- %s Thread[%3d] - Tensor<%d> : ", target, tid, m);
    for (int i = 0; i < m; i++) {
      printf("%3.0f ", __half2float(ptr[i]));
    }
    printf("\n");
  }
}

__host__ __device__ void printme(__half *ptr, int m, int n,
                                 int printingThread = 0, int until_n = 0,
                                 int until_m = 0, char *name = nullptr) {
  int tid = 0;
  if (until_n == 0) until_n = n;
  if (until_m == 0) until_m = m;
#if defined(__CUDA_ARCH__)
  tid = threadIdx.x;
  char * target = "GPU";
#else
  char * target = "CPU";
#endif 
  if (tid == printingThread) {
    printf("==-- %s Thread[%3d] - Tensor<%dx%d> : ", target, tid, m, n);
    name ? printf(" [%s]---===\n", name) : printf("---===\n");

    printf("\t");
    for (int i = 0; i < until_n; ++i) {
      printf("%3d ", i);
    }
    if (n > until_n) {
      printf("... %3d ", (n - 1));
    }
    printf("\n\t");
    for (int i = 0; i < sz16; ++i) {
      printf("__ ");
    }
    printf("\n\n");

    for (int j = 0; j < until_m; ++j) {
      printf("[%2d]  ", j);
      for (int i = 0; i < until_n; ++i) {
        __half val = ptr[j * n + i];
        printf("%3.0f ", __half2float(val));
      }
      printf("\n");
    }
    if (m > until_m) {
      printf("..\n[%2d]", (m - 1));
    }
    printf("\n");
  }
}

__host__ __device__ void printme(float *ptr, int m, int n,
                                 int printingThread = 0, int until_n = 0,
                                 int until_m = 0, char *name = nullptr) {
  int tid = 0;
  if (until_n == 0) until_n = n;
  if (until_m == 0) until_m = m;
#if defined(__CUDA_ARCH__)
  tid = threadIdx.x;
  char * target = "GPU";
#else
  char * target = "CPU";
#endif 
  if (tid == printingThread) {
    printf("===--- %s Thread[%d] - Tensor<%dx%d>", target, tid, m, n);
    name ? printf(" [%s]---===\n", name) : printf("---===\n");

    printf("\t");
    for (int i = 0; i < n; ++i) {
      printf("%3d ", i);
    }
    if (n > until_n) {
      printf("... %3d ", (n - 1));
    }
    printf("\n\t");
    for (int i = 0; i < sz16; ++i) {
      printf("__ ");
    }
    printf("\n\n");

    for (int j = 0; j < until_m; ++j) {
      printf("[%2d]  ", j);
      for (int i = 0; i < until_n; ++i) {
        float val = ptr[j * n + i];
        printf("%3.0f ", val);
      }
      printf("\n");
    }
    if (m > until_m) {
      printf("..\n[%2d]", (m - 1));
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
    uint16_t start_address_ : 14, : 2;  // 14 bits [0,14), 2 bits unused
    // leading dimension byte offset, bit [16,30), 4LSB not included
    // For N: This is the stride from the first col to the second col of the 8x2
    // brick in INTERLEAVED
    //   Unused for all SWIZZLE_* layouts (and assumed to be 1)
    // For T: This is the stride from the first 8 rows to the next 8 rows.
    uint16_t leading_byte_offset_ : 14, : 2;  // 14 bits [0,14), 2 bits unused
    // stride dimension byte offset, bit [32,46), 4LSB not included
    // For N: This is the stride from the first 8 rows to the next 8 rows.
    // For T: This is the stride fro mthe first 8 cols to the next 8 cols.
    uint16_t stride_byte_offset_ : 14, : 2;  // 14 bits [0,14), 2 bits unused
    // base_offset, bit [49,52)
    // Valid only for SWIZZLE_128B and SWIZZLE_64B
    uint8_t : 1,
        base_offset_ : 3, : 4;  // 1 bit unused, 3 bits [1,4), 4 bits unused
    // layout type, bit [62,64)
    // SWIZZLE_NONE = 0, SWIZZLE_32B = 3, SWIZZLE_64B = 2, SWIZZLE_128B = 1
    uint8_t : 6, layout_type_ : 2;  // 6 bits unused, 2 bits [6,8)
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
inline __device__ void fma(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t &d00,
    uint32_t &d01, uint32_t &d02, uint32_t &d03, uint32_t &d04, uint32_t &d05,
    uint32_t &d06, uint32_t &d07, uint32_t &d08, uint32_t &d09, uint32_t &d10,
    uint32_t &d11, uint32_t &d12, uint32_t &d13, uint32_t &d14, uint32_t &d15,
    uint32_t &d16, uint32_t &d17, uint32_t &d18, uint32_t &d19, uint32_t &d20,
    uint32_t &d21, uint32_t &d22, uint32_t &d23, uint32_t &d24, uint32_t &d25,
    uint32_t &d26, uint32_t &d27, uint32_t &d28, uint32_t &d29, uint32_t &d30,
    uint32_t &d31) {
  // The operation of the form D = A*B is issued when the input predicate
  // argument scale-d is false.
  constexpr int scale_D = 0;  // `p` in PTX

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
  constexpr int tnspB = 1;  // Switch to row-major
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
#ifdef DEBUG2
  if (threadIdx.x == 0) {
    printf("[%d] Desc_A: 0x%016" PRIx64 " Desc_B: 0x%016" PRIx64 "\n",
           threadIdx.x, desc_a, desc_b);
  }
#endif
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
  constexpr int scale_D = 0;  // `p` in PTX

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
  constexpr int tnspB = 1;  // Switch to row-major
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

template <int N>
inline __device__ void warpgroup_wait() {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900)
  static_assert(N >= 0 && N <= 7,
                "_warpgroup.wait {N}; must be in range [0, 7]");
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
#endif
}

inline __device__ void warpgroup_fence_operand(uint32_t &reg) {
#if defined(__CUDA_ARCH__)
  asm volatile("" : "+r"(reg)::"memory");
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

template <class T>
inline __device__ void fillme(T *src, T *dst, int n, int m) {
  for (int i = threadIdx.x; i < n * m; i += blockDim.x) dst[i] = src[i];
}

template <class ElementA, class ElementB, class ElementC>
__global__ void test(ElementA *lhs, ElementB *rhs, ElementC *acc) {
  /////////////////////////////////////////////////////////////////////////////////
  // Prologue
  extern __shared__ char smem[];

  ElementA *smem_a_ptr = reinterpret_cast<ElementA *>(smem);
  ElementB *smem_b_ptr =
      reinterpret_cast<ElementB *>(smem + (sz64 * sz16 * sizeof(ElementC)));

  if(threadIdx.x == 0) {
    for(int i= 0 ; i < 16; ++i) {
      ElementA *d1 = &lhs[i * sz64];
      int off = (i % 2 ? core8 : 0) + ((i/2) * 8 * 8 * 2);
      ElementA *s1 = &smem_a_ptr[off];
      // Core matrix copy 8 x 8
      for(int t=0; t < core8; ++t) {
        ElementA *d = &d1[t * core8];
        ElementA *s = &s1[t * 16];
        for(int e = 0; e < 8; ++e) {
          s[e] = d[e];
        }
      }
    }
  }
  fillme(lhs, smem_a_ptr, sz64, sz16);
  fillme(rhs, smem_b_ptr, sz16, sz128);

  __syncthreads();

#ifdef DEBUG1
  printme(lhs, sz64, sz16, 0, 0,0, "Global Memory");
  printme(rhs, sz16, sz128, 0,0,0,"Global Memory");

  printme(smem_a_ptr, sz64, sz16, 0, 0,0, "Shared Memory");
  printme(smem_b_ptr, sz16, sz128, 0,0,0,"Shared Memory");
#endif

  /////////////////////////////////////////////////////////////////////////////////
  // GEMM

  // Create Matrix descriptors
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor

  GmmaDescriptor desc_a;
  desc_a.start_address_ = cast_smem_ptr_to_uint(smem_a_ptr) >> 4;
  desc_a.leading_byte_offset_ = uint32_t(16);
  desc_a.stride_byte_offset_ = uint32_t(256);
  desc_a.base_offset_ = 0;
  desc_a.layout_type_ = uint8_t(LayoutType::INTERLEAVE);

  GmmaDescriptor desc_b;
  desc_b.start_address_ = (cast_smem_ptr_to_uint(smem_b_ptr) >> 4);
  desc_b.leading_byte_offset_ = uint32_t(2048);
  desc_b.stride_byte_offset_ = uint32_t(16);
  desc_b.base_offset_ = 0;
  desc_b.layout_type_ = uint8_t(LayoutType::INTERLEAVE);

#ifdef DEBUG2
  if (threadIdx.x == 0) {
    printf("%p %p \n", smem_a_ptr, smem_b_ptr);
    print(desc_a);
    print(desc_b);
  }
#endif

  // Result registers
  ElementC regs[sz64] = {ElementC(0)};
#ifndef F32
  uint32_t *r = reinterpret_cast<uint32_t *>(&regs);
#else
  float *r = reinterpret_cast<float *>(&regs);
#endif

  warpgroup_fence_operand(r[0]);
  warpgroup_arrive();
#ifndef F32
  // F16 = F16 * F16 + F16
  fma(desc_a, desc_b, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8],
      r[9], r[10], r[11], r[12], r[13], r[14], r[15], r[16], r[17], r[18],
      r[19], r[20], r[21], r[22], r[23], r[24], r[25], r[26], r[27], r[28],
      r[29], r[30], r[31]);
#else
  // F32 = F16 * F16 + F32
  fma(desc_a, desc_b, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8],
      r[9], r[10], r[11], r[12], r[13], r[14], r[15], r[16], r[17], r[18],
      r[19], r[20], r[21], r[22], r[23], r[24], r[25], r[26], r[27], r[28],
      r[29], r[30], r[31], r[32], r[33], r[34], r[35], r[36], r[37], r[38],
      r[39], r[40], r[41], r[42], r[43], r[44], r[45], r[46], r[47], r[48],
      r[49], r[50], r[51], r[52], r[53], r[54], r[55], r[56], r[57], r[58],
      r[59], r[60], r[61], r[62], r[63]);
#endif

#if 0
  // Swizzling (cutlass does something similar to this)

  // Matrix-LHS
  // start_addr :  0x0040
  // leading_off:  0x0001 (1)
  // stride_off :  0x0040 (64)
  // base_offset:  0x0
  // layout_type:  0x1 (B128)

  // Matrix-RHS
  // GmmaDescriptor: 0x4000008000401c40
  // start_addr :  0x1c40
  // leading_off:  0x0040 (64)
  // stride_off :  0x0080 (128)
  // base_offset:  0x0
  // layout_type:  0x1 (B128)

  // Starts : Desc_A: 0x4000004000010040 Desc_B: 0x4000008000401c40

  // [iter=0] Desc_A: 0x4000004000010040 Desc_B: 0x4000008000401c40
  // [iter=1] Desc_A: 0x4000004000010240 Desc_B: 0x4000008000401c40
  // [iter=2] Desc_A: 0x4000004000010042 Desc_B: 0x4000008000401d40
  // [iter=3] Desc_A: 0x4000004000010242 Desc_B: 0x4000008000401d40
  // [iter=4] Desc_A: 0x4000004000010044 Desc_B: 0x4000008000401e40
  // [iter=5] Desc_A: 0x4000004000010244 Desc_B: 0x4000008000401e40
  // [iter=6] Desc_A: 0x4000004000010046 Desc_B: 0x4000008000401f40
  // [iter=7] Desc_A: 0x4000004000010246 Desc_B: 0x4000008000401f40

  // Increments
  // [iter=0] Desc_A        | Desc_B
  // [iter=1] Desc_A + 512  | Desc_B
  // [iter=2] Desc_A + 2    | Desc_B + 256
  // [iter=3] Desc_A + 514  | Desc_B + 256
  // [iter=4] Desc_A + 4    | Desc_B + 512
  // [iter=5] Desc_A + 516  | Desc_B + 512
  // [iter=6] Desc_A + 6    | Desc_B + 768
  // [iter=7] Desc_A + 518  | Desc_B + 768

  // Iteration 0
  fma(desc_a, desc_b, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8],
      r[9], r[10], r[11], r[12], r[13], r[14], r[15], r[16], r[17], r[18],
      r[19], r[20], r[21], r[22], r[23], r[24], r[25], r[26], r[27], r[28],
      r[29], r[30], r[31]);
  // Iteration 1
  fma(desc_a + uint64_t(512), desc_b, r[0], r[1], r[2], r[3], r[4], r[5], r[6],
      r[7], r[8], r[9], r[10], r[11], r[12], r[13], r[14], r[15], r[16], r[17],
      r[18], r[19], r[20], r[21], r[22], r[23], r[24], r[25], r[26], r[27],
      r[28], r[29], r[30], r[31]);
  // Iteration 2
  fma(desc_a + uint64_t(2), desc_b + uint64_t(256), r[0], r[1], r[2], r[3],
      r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13], r[14],
      r[15], r[16], r[17], r[18], r[19], r[20], r[21], r[22], r[23], r[24],
      r[25], r[26], r[27], r[28], r[29], r[30], r[31]);
  // Iteration 3
  fma(desc_a + uint64_t(514), desc_b + uint64_t(256), r[0], r[1], r[2], r[3],
      r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13], r[14],
      r[15], r[16], r[17], r[18], r[19], r[20], r[21], r[22], r[23], r[24],
      r[25], r[26], r[27], r[28], r[29], r[30], r[31]);
  // Iteration 4
  fma(desc_a + uint64_t(4), desc_b + uint64_t(512), r[0], r[1], r[2], r[3],
      r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13], r[14],
      r[15], r[16], r[17], r[18], r[19], r[20], r[21], r[22], r[23], r[24],
      r[25], r[26], r[27], r[28], r[29], r[30], r[31]);
  // Iteration 5
  fma(desc_a + uint64_t(516), desc_b + uint64_t(512), r[0], r[1], r[2], r[3],
      r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13], r[14],
      r[15], r[16], r[17], r[18], r[19], r[20], r[21], r[22], r[23], r[24],
      r[25], r[26], r[27], r[28], r[29], r[30], r[31]);
  // Iteration 6
  fma(desc_a + uint64_t(6), desc_b + uint64_t(768), r[0], r[1], r[2], r[3],
      r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13], r[14],
      r[15], r[16], r[17], r[18], r[19], r[20], r[21], r[22], r[23], r[24],
      r[25], r[26], r[27], r[28], r[29], r[30], r[31]);
  // Iteration 7
  fma(desc_a + uint64_t(518), desc_b + uint64_t(768), r[0], r[1], r[2], r[3],
      r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13], r[14],
      r[15], r[16], r[17], r[18], r[19], r[20], r[21], r[22], r[23], r[24],
      r[25], r[26], r[27], r[28], r[29], r[30], r[31]);
#endif

  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(r[0]);

#ifdef DEBUG3
  if(threadIdx.x==0)
    printf("\nProduct of some threads after wgmma.mma_async\n");
  printme1(regs, sz64, 0);
  __syncthreads();
  printme1(regs, sz64, 1);
  __syncthreads();
  printme1(regs, sz64, 16);
  __syncthreads();
  printme1(regs, sz64, 32);
  __syncthreads();
  printme1(regs, sz64, 48);
  __syncthreads();
  printme1(regs, sz64, 64);
  __syncthreads();
  printme1(regs, sz64, 80);
  __syncthreads();
  printme1(regs, sz64, 96);
  __syncthreads();
  printme1(regs, sz64, 112);
  __syncthreads();
#endif

  /////////////////////////////////////////////////////////////////////////////////
  // Epilogue
  for (int i = 0; i < sz64; i++) {
    acc[i * sz128 + threadIdx.x] = regs[i];
  }
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template <typename T>
T *malloc_managed(size_t n, T init = T()) {
  T *data;
  bool failure =
      cudaMallocManaged((void **)&data, sizeof(T) * n) != cudaSuccess;
  if (failure) {
    printf("Setup failed\n");
    exit(-1);
  }
  return data;
}

template <typename T>
T *malloc_managed() {
  return malloc_managed<T>(1);
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template <class ElementA, class ElementB, class ElementC>
__global__ void reference(ElementA *lhs, ElementB *rhs, ElementC *acc) {
  for (int i = 0; i < sz64; i++) {
    for (int j = threadIdx.x; j < sz128; j += blockDim.x) {
      for (int k = 0; k < sz16; ++k) {
        acc[i * sz128 + j] += ElementC(lhs[i * sz16 + k] * rhs[k * sz128 + j]);
      }
    }
  }
}

template <class ElementC>
bool verify(ElementC *acc, ElementC *ref_acc) {
  bool passed = true;
  for (int i = 0; i < sz64; i++) {
    for (int j = 0; j < sz128; j++) {
      if (__half2float(acc[i * sz128 + j]) !=
          __half2float(ref_acc[i * sz128 + j])) {
        passed = false;
      }
    }
  }
  printf("Test %s\n", passed ? "PASSED" : "FAILED");
  return passed;
}

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

int main() {
  // Allocate memory
  __half *lhs = malloc_managed<__half>(sz64 * sz16);
  __half *rhs = malloc_managed<__half>(sz128 * sz16);
#ifdef F32
  using ElementC = float;
#else
  using ElementC = __half;
#endif
  ElementC *acc = malloc_managed<ElementC>(sz128 * sz64);
  ElementC *ref_acc = malloc_managed<ElementC>(sz128 * sz64);
  for (int i = 0; i < sz128 * sz64; i++) {
    ref_acc[i] = ElementC(0);
    acc[i] = ElementC(0);
  }

  // Initialize data
  for (int i = 0; i < sz64; i++) {
    for (int j = 0; j < sz16; j++) {
      // lhs[i * sz16 + j] = __float2half((j+i*10)%30);
      lhs[i * sz16 + j] = __float2half((i * sz16 + j) % 32);
      // lhs[i * sz16 + j] = __float2half(2);
    }
  }
  for (int i = 0; i < sz16; i++)
    for (int j = 0; j < sz128; j++) {
      // rhs[i * sz128 + j] = __float2half((j+i)%30);      
      rhs[i * sz128 + j] = __float2half(1);
    }

#ifdef DEBUG5
  printme(lhs, sz64, sz16);
  printme(rhs, sz16, sz128);
#endif

  // Luanch a kernel for warpgroup level GEMM
  int smem_size = 227 << 10;
  gpuErrchk(cudaFuncSetAttribute(test<__half, __half, ElementC>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem_size));
  gpuErrchk(cudaFuncSetAttribute(test<__half, __half, ElementC>,
                                 cudaFuncAttributePreferredSharedMemoryCarveout,
                                 100));

  test<<<1, 128, smem_size>>>(lhs, rhs, acc);
  gpuErrchk(cudaPeekAtLastError());
  cudaDeviceSynchronize();
  gpuErrchk(cudaPeekAtLastError());

  reference<<<1, 1>>>(lhs, rhs, ref_acc);
  gpuErrchk(cudaPeekAtLastError());
  cudaDeviceSynchronize();
  gpuErrchk(cudaPeekAtLastError());



  if (!verify(ref_acc, acc)) {
    printme(ref_acc, sz64, sz128, 0, 32, 4, "CPU REFERENCE");
    printme(acc, sz64, sz128, 0, 32, 4, "GPU RESULT");
  }

  return 0;
}
