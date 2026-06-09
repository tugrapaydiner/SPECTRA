// SPECTRA fused sparse ternary GEMV kernel (BLUEPRINT sections 12, 26.3).
//
// Strictly optimized for B=1 edge inference, where the operation is memory-bound
// GEMV (matrix-vector), not GEMM. The kernel fuses, in ONE cache-hot streaming
// pass per active token:
//   1. active-token gather   (lazy routing skips frozen tokens)
//   2. vectorized 2-bit -> int8 ternary unpack via AVX2 pshufb LUT
//   3. ternary GEMV via _mm256_sign_epi8 (no FP MACs)
//   4. INTEGER multiply-shift requantize (no FP division in the hot loop)
//   5. scatter the int8 result back to the active token
//
// This replaces the previous reference that (a) decoded ternary with a SCALAR
// inner loop and (b) requantized with floating-point division -- both of which
// destroyed the bandwidth/throughput the design depends on. The scalar path is
// kept as an exact correctness oracle; the AVX2 path is validated bit-for-bit
// against it AND against the PyTorch fake-quant reference (tests/test_kernel.py).

#include <cstdint>
#include <cstddef>
#include <cstdlib>  // malloc/free -- keep the kernel free of any C++ runtime deps

#if defined(__AVX2__)
#include <immintrin.h>
#endif

extern "C" {

// Decode a 2-bit ternary code (00->0, 01->+1, 10->-1, 11->reserved 0).
static inline int32_t decode_ternary_scalar(uint8_t code) {
  switch (code & 0x3) {
    case 0x1: return 1;
    case 0x2: return -1;
    default:  return 0;
  }
}

// Integer multiply-shift requantize: int32 accumulator -> int8 output.
//   y = clamp( round( acc * mult / 2^shift ), -128, 127 )
// `mult` folds (w_scale[o] * act_scale / out_scale) into a fixed-point integer.
static inline int8_t requantize_i32(int32_t acc, int32_t mult, int32_t shift) {
  const int64_t rounding = (int64_t)1 << (shift - 1);
  int64_t v = ((int64_t)acc * (int64_t)mult + rounding) >> shift;
  if (v > 127) v = 127;
  if (v < -128) v = -128;
  return (int8_t)v;
}

// Dot product of an int8 activation row with one packed-ternary weight row.
// w_packed holds `hidden_dim` 2-bit codes (4 per byte). Returns int32 accumulator.
static int32_t ternary_dot(const int8_t* x, const uint8_t* w_packed, int hidden_dim) {
  int32_t acc = 0;
  int d = 0;

#if defined(__AVX2__)
  // --- pshufb decode LUT: code (low 2 bits) -> {-1,0,+1} (int8), both lanes. ---
  const __m256i lut = _mm256_setr_epi8(
      0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  // Replicate control: lane j -> source byte (j>>2) within its 128-bit half.
  const __m256i rep = _mm256_setr_epi8(
      0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
      4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7);
  // Per-sub-position lane selectors (lane%4 == q) as packed 32-bit masks.
  const __m256i m0 = _mm256_set1_epi32(0x000000FF);
  const __m256i m1 = _mm256_set1_epi32(0x0000FF00);
  const __m256i m2 = _mm256_set1_epi32(0x00FF0000);
  const __m256i m3 = _mm256_set1_epi32((int)0xFF000000);
  const __m256i lo2 = _mm256_set1_epi8(0x03);

  __m256i vacc = _mm256_setzero_si256();
  for (; d + 32 <= hidden_dim; d += 32) {
    // Load 8 packed bytes (= 32 ternary codes); broadcast to both 128-bit halves.
    __m128i p8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w_packed + (d >> 2)));
    __m256i bcast = _mm256_set_m128i(p8, p8);
    __m256i raw = _mm256_shuffle_epi8(bcast, rep);  // lane j = byte(j>>2)

    // Extract sub-position q in {0,1,2,3} via 16-bit right shifts, then select the
    // correct shift per lane with the static masks (mask q -> lanes lane%4==q).
    __m256i s0 = raw;
    __m256i s1 = _mm256_srli_epi16(raw, 2);
    __m256i s2 = _mm256_srli_epi16(raw, 4);
    __m256i s3 = _mm256_srli_epi16(raw, 6);
    __m256i codes = _mm256_or_si256(
        _mm256_or_si256(_mm256_and_si256(s0, m0), _mm256_and_si256(s1, m1)),
        _mm256_or_si256(_mm256_and_si256(s2, m2), _mm256_and_si256(s3, m3)));
    codes = _mm256_and_si256(codes, lo2);              // isolate the 2-bit code
    __m256i w = _mm256_shuffle_epi8(lut, codes);       // decode -> int8 {-1,0,1}

    // Ternary GEMV: signed(x) by the ternary weight, widen, accumulate (int32).
    __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + d));
    __m256i prod = _mm256_sign_epi8(xv, w);            // +x / -x / 0
    __m256i lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(prod));
    __m256i hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(prod, 1));
    vacc = _mm256_add_epi32(vacc, _mm256_madd_epi16(lo, _mm256_set1_epi16(1)));
    vacc = _mm256_add_epi32(vacc, _mm256_madd_epi16(hi, _mm256_set1_epi16(1)));
  }
  alignas(32) int32_t tmp[8];
  _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), vacc);
  for (int k = 0; k < 8; ++k) acc += tmp[k];
#endif

  // Scalar tail (and the entire scalar reference build when AVX2 is disabled).
  for (; d < hidden_dim; ++d) {
    const uint8_t byte = w_packed[d >> 2];
    acc += decode_ternary_scalar((byte >> (2 * (d & 0x3))) & 0x3) * (int32_t)x[d];
  }
  return acc;
}

// Dot of an int8 activation with an ALREADY-DECODED int8 ternary weight row
// (skips the 2-bit unpack -- used when a row is decoded once and reused K times).
static int32_t ternary_dot_decoded(const int8_t* x, const int8_t* w, int hidden_dim) {
  int32_t acc = 0;
  int d = 0;
#if defined(__AVX2__)
  __m256i vacc = _mm256_setzero_si256();
  for (; d + 32 <= hidden_dim; d += 32) {
    __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + d));
    __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + d));
    __m256i prod = _mm256_sign_epi8(xv, wv);
    __m256i lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(prod));
    __m256i hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(prod, 1));
    vacc = _mm256_add_epi32(vacc, _mm256_madd_epi16(lo, _mm256_set1_epi16(1)));
    vacc = _mm256_add_epi32(vacc, _mm256_madd_epi16(hi, _mm256_set1_epi16(1)));
  }
  alignas(32) int32_t tmp[8];
  _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), vacc);
  for (int k = 0; k < 8; ++k) acc += tmp[k];
#endif
  for (; d < hidden_dim; ++d) acc += static_cast<int32_t>(w[d]) * static_cast<int32_t>(x[d]);
  return acc;
}

static void decode_row(const uint8_t* w_packed, int8_t* out, int hidden_dim) {
  for (int d = 0; d < hidden_dim; ++d)
    out[d] = static_cast<int8_t>(decode_ternary_scalar((w_packed[d >> 2] >> (2 * (d & 0x3))) & 0x3));
}

// Weight-stationary recursion GEMV (the only way to beat the B=1 roofline).
//
// A single B=1 GEMV is memory-bound (AI = 4 ops/byte): each weight is read once.
// But recursion re-applies the SAME ternary matrix K = T*n*N_sup times. This
// kernel exploits that: it decodes each weight row ONCE and dots it against all K
// recursion-step activation vectors before moving on -- the row stays in L1, the
// 2-bit unpack is amortised over K, and DRAM pays for the matrix once. Effective
// arithmetic intensity rises to ~4K, pushing the kernel compute-bound.
//
//   X  : [K, hidden]   the K recursion-step activations (weight-stationary inner)
//   Y  : [K, out_dim]  outputs
void spectra_weight_stationary_gemv(
    const int8_t* X, const uint8_t* W_packed, const int32_t* requant_mult,
    int requant_shift, int K, int hidden_dim, int out_dim, int8_t* Y) {
  const int packed_row = hidden_dim / 4;
  int8_t* wrow = (int8_t*)malloc((size_t)hidden_dim);
  if (!wrow) return;
  for (int o = 0; o < out_dim; ++o) {
    decode_row(W_packed + (size_t)o * packed_row, wrow, hidden_dim);  // decode ONCE
    const int32_t m = requant_mult[o];
    for (int k = 0; k < K; ++k) {                                     // reuse across K
      const int32_t acc = ternary_dot_decoded(X + (size_t)k * hidden_dim, wrow, hidden_dim);
      Y[(size_t)k * out_dim + o] = requantize_i32(acc, m, requant_shift);
    }
  }
  free(wrow);
}

// Fused sparse ternary GEMV for B=1.
//
//   X            : [num_tokens, hidden_dim] int8 activations (row-major)
//   active_idx   : indices of the tokens to compute (lazy routing)
//   num_active   : number of active tokens
//   W_packed     : [out_dim, hidden_dim/4] packed 2-bit ternary weights
//   requant_mult : [out_dim] per-output-channel fixed-point requant multiplier
//   requant_shift: scalar right-shift for the requant (e.g. 15)
//   hidden_dim   : input width (multiple of 4)
//   out_dim      : output width
//   Y            : [num_tokens, out_dim] int8 output (scatter target)
void spectra_sparse_ternary_gemv(
    const int8_t* X, const int32_t* active_idx, int num_active,
    const uint8_t* W_packed, const int32_t* requant_mult, int requant_shift,
    int hidden_dim, int out_dim, int8_t* Y) {
  const int packed_row = hidden_dim / 4;  // bytes per weight row
  for (int t = 0; t < num_active; ++t) {
    const int token = active_idx[t];
    const int8_t* x = X + (size_t)token * hidden_dim;  // gather (cache-hot)
    int8_t* y = Y + (size_t)token * out_dim;           // scatter target
    for (int o = 0; o < out_dim; ++o) {
      const uint8_t* w = W_packed + (size_t)o * packed_row;
      const int32_t acc = ternary_dot(x, w, hidden_dim);
      y[o] = requantize_i32(acc, requant_mult[o], requant_shift);
    }
  }
}

// Fused two-layer ternary FFN for B=1: up-projection -> int8 ReLU -> down-
// projection, entirely in C++. Lets the deep recursion call ONE native function
// per block instead of returning to Python between matmuls (the B=1 overhead
// killer). The int8 intermediate stays cache-hot between the two GEMVs.
void spectra_fused_ternary_ffn(
    const int8_t* X, const int32_t* active_idx, int num_active,
    const uint8_t* W1_packed, const int32_t* mult1,
    const uint8_t* W2_packed, const int32_t* mult2,
    int requant_shift, int hidden_dim, int inter_dim, int out_dim, int8_t* Y) {
  const int p1 = hidden_dim / 4;   // packed bytes per W1 row
  const int p2 = inter_dim / 4;    // packed bytes per W2 row
  int8_t* tmp = (int8_t*)malloc((size_t)inter_dim);  // cache-hot int8 intermediate
  if (!tmp) return;
  for (int t = 0; t < num_active; ++t) {
    const int token = active_idx[t];
    const int8_t* x = X + (size_t)token * hidden_dim;
    // Up-projection + fused int8 ReLU.
    for (int o = 0; o < inter_dim; ++o) {
      const int32_t acc = ternary_dot(x, W1_packed + (size_t)o * p1, hidden_dim);
      const int8_t v = requantize_i32(acc, mult1[o], requant_shift);
      tmp[o] = v > 0 ? v : 0;
    }
    // Down-projection consumes the cache-hot intermediate (no Python in between).
    int8_t* y = Y + (size_t)token * out_dim;
    for (int o = 0; o < out_dim; ++o) {
      const int32_t acc = ternary_dot(tmp, W2_packed + (size_t)o * p2, inter_dim);
      y[o] = requantize_i32(acc, mult2[o], requant_shift);
    }
  }
  free(tmp);
}

}  // extern "C"
