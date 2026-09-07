// SPECTRA fused sparse ternary GEMV kernel.
// Milestone 02 numerical/input contract:
//   * activations: full signed INT8 range [-128, 127]
//   * weights: ternary {-1,0,+1}, packed 4 x 2-bit codes per byte
//   * row packing: ceil(hidden_dim/4) bytes per output row; padding codes are zero
//   * code 11 is reserved and rejected by checked public entry points
//   * requant shift: integer in [0, 62]
//   * multiplier: non-negative int32 fixed-point multiplier
//   * hidden/inter dimensions: <= floor(INT32_MAX / 128), so int32 dot sums cannot overflow
//
// Public C entry points are checked and return a SpectraStatus. The AVX2 dot
// keeps the fast byte-sign path, detects the one overflowing case INT8_MIN * -1,
// and adds the exact +256 correction needed to recover the mathematical +128.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <climits>
#include <limits>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

extern "C" {

enum SpectraStatus : int {
  SPECTRA_OK = 0,
  SPECTRA_NULL_POINTER = 1,
  SPECTRA_INVALID_DIMENSION = 2,
  SPECTRA_INVALID_SHIFT = 3,
  SPECTRA_INVALID_LENGTH = 4,
  SPECTRA_ACTIVE_INDEX_OOB = 5,
  SPECTRA_INVALID_PACKED_WEIGHT = 6,
  SPECTRA_INVALID_MULTIPLIER = 7,
  SPECTRA_ALLOCATION_FAILURE = 8,
};

const char* spectra_status_string(int status) {
  switch (status) {
    case SPECTRA_OK: return "ok";
    case SPECTRA_NULL_POINTER: return "null pointer";
    case SPECTRA_INVALID_DIMENSION: return "invalid dimension";
    case SPECTRA_INVALID_SHIFT: return "invalid requantization shift";
    case SPECTRA_INVALID_LENGTH: return "invalid buffer length";
    case SPECTRA_ACTIVE_INDEX_OOB: return "active index out of bounds";
    case SPECTRA_INVALID_PACKED_WEIGHT: return "invalid packed ternary weight/padding";
    case SPECTRA_INVALID_MULTIPLIER: return "invalid requantization multiplier";
    case SPECTRA_ALLOCATION_FAILURE: return "native allocation failure";
    default: return "unknown SPECTRA status";
  }
}

int spectra_compiled_with_avx2() {
#if defined(__AVX2__)
  return 1;
#else
  return 0;
#endif
}

static constexpr int kMaxDotWidth = INT32_MAX / 128;

static inline size_t packed_row_bytes(int hidden_dim) {
  return (static_cast<size_t>(hidden_dim) + 3u) / 4u;
}

static bool checked_product(size_t a, size_t b, size_t* out) {
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
  *out = a * b;
  return true;
}

static inline int32_t decode_ternary_scalar(uint8_t code) {
  switch (code & 0x3u) {
    case 0x1u: return 1;
    case 0x2u: return -1;
    default: return 0;
  }
}

static bool valid_packed_rows(const uint8_t* packed, int rows, int hidden_dim) {
  const size_t row_bytes = packed_row_bytes(hidden_dim);
  const int tail_codes = hidden_dim & 3;
  for (int r = 0; r < rows; ++r) {
    const uint8_t* row = packed + static_cast<size_t>(r) * row_bytes;
    for (size_t b = 0; b < row_bytes; ++b) {
      const uint8_t byte = row[b];
      const int n_codes = (b + 1 == row_bytes && tail_codes != 0) ? tail_codes : 4;
      for (int q = 0; q < n_codes; ++q) {
        if (((byte >> (2 * q)) & 0x3u) == 0x3u) return false;
      }
      for (int q = n_codes; q < 4; ++q) {
        if (((byte >> (2 * q)) & 0x3u) != 0u) return false;
      }
    }
  }
  return true;
}

static inline int8_t requantize_i32(int32_t acc, int32_t mult, int32_t shift) {
  const int64_t product = static_cast<int64_t>(acc) * static_cast<int64_t>(mult);
  const int64_t rounding = shift == 0 ? 0 : (static_cast<int64_t>(1) << (shift - 1));
  int64_t v = shift == 0 ? product : ((product + rounding) >> shift);
  if (v > 127) v = 127;
  if (v < -128) v = -128;
  return static_cast<int8_t>(v);
}

#if defined(__AVX2__)
static inline uint32_t popcount32_portable(uint32_t x) {
  x = x - ((x >> 1) & 0x55555555u);
  x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
  x = (x + (x >> 4)) & 0x0F0F0F0Fu;
  return (x * 0x01010101u) >> 24;
}
#endif

static int32_t ternary_dot(const int8_t* x, const uint8_t* w_packed, int hidden_dim) {
  int32_t acc = 0;
  int d = 0;

#if defined(__AVX2__)
  const __m256i lut = _mm256_setr_epi8(
      0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  const __m256i rep = _mm256_setr_epi8(
      0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
      4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7);
  const __m256i m0 = _mm256_set1_epi32(0x000000FF);
  const __m256i m1 = _mm256_set1_epi32(0x0000FF00);
  const __m256i m2 = _mm256_set1_epi32(0x00FF0000);
  const __m256i m3 = _mm256_set1_epi32(static_cast<int>(0xFF000000u));
  const __m256i lo2 = _mm256_set1_epi8(0x03);
  const __m256i ones16 = _mm256_set1_epi16(1);
  const __m256i min8 = _mm256_set1_epi8(static_cast<char>(0x80));
  const __m256i neg1_8 = _mm256_set1_epi8(-1);
  uint64_t correction = 0;
  __m256i vacc = _mm256_setzero_si256();

  for (; d + 32 <= hidden_dim; d += 32) {
    const __m128i p8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w_packed + (d >> 2)));
    const __m256i bcast = _mm256_set_m128i(p8, p8);
    const __m256i raw = _mm256_shuffle_epi8(bcast, rep);
    const __m256i s0 = raw;
    const __m256i s1 = _mm256_srli_epi16(raw, 2);
    const __m256i s2 = _mm256_srli_epi16(raw, 4);
    const __m256i s3 = _mm256_srli_epi16(raw, 6);
    __m256i codes = _mm256_or_si256(
        _mm256_or_si256(_mm256_and_si256(s0, m0), _mm256_and_si256(s1, m1)),
        _mm256_or_si256(_mm256_and_si256(s2, m2), _mm256_and_si256(s3, m3)));
    codes = _mm256_and_si256(codes, lo2);
    const __m256i w8 = _mm256_shuffle_epi8(lut, codes);
    const __m256i x8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + d));

    // VPSIGNB is correct for every ternary product except INT8_MIN * -1:
    // the 8-bit negate wraps to INT8_MIN, exactly 256 below the true +128.
    // Keep the fast byte-sign path and add a +256 correction for those lanes.
    const __m256i prod8 = _mm256_sign_epi8(x8, w8);
    const __m256i special = _mm256_and_si256(
        _mm256_cmpeq_epi8(x8, min8), _mm256_cmpeq_epi8(w8, neg1_8));
    correction += static_cast<uint64_t>(popcount32_portable(
        static_cast<uint32_t>(_mm256_movemask_epi8(special)))) * 256u;
    const __m256i p_lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(prod8));
    const __m256i p_hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(prod8, 1));
    vacc = _mm256_add_epi32(vacc, _mm256_madd_epi16(p_lo16, ones16));
    vacc = _mm256_add_epi32(vacc, _mm256_madd_epi16(p_hi16, ones16));
  }

  alignas(32) int32_t tmp[8];
  _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), vacc);
  for (int k = 0; k < 8; ++k) acc += tmp[k];
  const int64_t vector_corrected = static_cast<int64_t>(acc) + static_cast<int64_t>(correction);
  acc = static_cast<int32_t>(vector_corrected);
#endif

  for (; d < hidden_dim; ++d) {
    const uint8_t byte = w_packed[d >> 2];
    acc += decode_ternary_scalar((byte >> (2 * (d & 0x3))) & 0x3u) * static_cast<int32_t>(x[d]);
  }
  return acc;
}

static int32_t ternary_dot_decoded(const int8_t* x, const int8_t* w, int hidden_dim) {
  int32_t acc = 0;
  int d = 0;
#if defined(__AVX2__)
  const __m256i ones16 = _mm256_set1_epi16(1);
  const __m256i min8 = _mm256_set1_epi8(static_cast<char>(0x80));
  const __m256i neg1_8 = _mm256_set1_epi8(-1);
  uint64_t correction = 0;
  __m256i vacc = _mm256_setzero_si256();
  for (; d + 32 <= hidden_dim; d += 32) {
    const __m256i x8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + d));
    const __m256i w8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + d));
    const __m256i prod8 = _mm256_sign_epi8(x8, w8);
    const __m256i special = _mm256_and_si256(
        _mm256_cmpeq_epi8(x8, min8), _mm256_cmpeq_epi8(w8, neg1_8));
    correction += static_cast<uint64_t>(popcount32_portable(
        static_cast<uint32_t>(_mm256_movemask_epi8(special)))) * 256u;
    const __m256i p_lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(prod8));
    const __m256i p_hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(prod8, 1));
    vacc = _mm256_add_epi32(vacc, _mm256_madd_epi16(p_lo16, ones16));
    vacc = _mm256_add_epi32(vacc, _mm256_madd_epi16(p_hi16, ones16));
  }
  alignas(32) int32_t tmp[8];
  _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), vacc);
  for (int k = 0; k < 8; ++k) acc += tmp[k];
  const int64_t vector_corrected = static_cast<int64_t>(acc) + static_cast<int64_t>(correction);
  acc = static_cast<int32_t>(vector_corrected);
#endif
  for (; d < hidden_dim; ++d) {
    acc += static_cast<int32_t>(w[d]) * static_cast<int32_t>(x[d]);
  }
  return acc;
}

static void decode_row(const uint8_t* w_packed, int8_t* out, int hidden_dim) {
  for (int d = 0; d < hidden_dim; ++d) {
    out[d] = static_cast<int8_t>(decode_ternary_scalar(
        (w_packed[d >> 2] >> (2 * (d & 0x3))) & 0x3u));
  }
}

static int validate_common(
    const int8_t* X, size_t x_len,
    const uint8_t* W_packed, size_t w_len,
    const int32_t* requant_mult, size_t mult_len,
    int requant_shift, int rows, int vectors, int hidden_dim,
    int8_t* Y, size_t y_len) {
  if (hidden_dim <= 0 || hidden_dim > kMaxDotWidth || rows <= 0 || vectors <= 0) {
    return SPECTRA_INVALID_DIMENSION;
  }
  if (requant_shift < 0 || requant_shift > 62) return SPECTRA_INVALID_SHIFT;
  size_t expected_x = 0, expected_w = 0, expected_y = 0;
  if (!checked_product(static_cast<size_t>(vectors), static_cast<size_t>(hidden_dim), &expected_x) ||
      !checked_product(static_cast<size_t>(rows), packed_row_bytes(hidden_dim), &expected_w) ||
      !checked_product(static_cast<size_t>(vectors), static_cast<size_t>(rows), &expected_y)) {
    return SPECTRA_INVALID_LENGTH;
  }
  if (x_len != expected_x || w_len != expected_w || mult_len != static_cast<size_t>(rows) || y_len != expected_y) {
    return SPECTRA_INVALID_LENGTH;
  }
  if (!X || !W_packed || !requant_mult || !Y) return SPECTRA_NULL_POINTER;
  if (!valid_packed_rows(W_packed, rows, hidden_dim)) return SPECTRA_INVALID_PACKED_WEIGHT;
  for (int r = 0; r < rows; ++r) {
    if (requant_mult[r] < 0) return SPECTRA_INVALID_MULTIPLIER;
  }
  return SPECTRA_OK;
}

int spectra_weight_stationary_gemv(
    const int8_t* X, size_t x_len,
    const uint8_t* W_packed, size_t w_len,
    const int32_t* requant_mult, size_t mult_len,
    int requant_shift, int K, int hidden_dim, int out_dim,
    int8_t* Y, size_t y_len) {
  const int status = validate_common(
      X, x_len, W_packed, w_len, requant_mult, mult_len,
      requant_shift, out_dim, K, hidden_dim, Y, y_len);
  if (status != SPECTRA_OK) return status;

  const size_t packed_row = packed_row_bytes(hidden_dim);
  int8_t* wrow = static_cast<int8_t*>(std::malloc(static_cast<size_t>(hidden_dim)));
  if (!wrow) return SPECTRA_ALLOCATION_FAILURE;
  for (int o = 0; o < out_dim; ++o) {
    decode_row(W_packed + static_cast<size_t>(o) * packed_row, wrow, hidden_dim);
    const int32_t m = requant_mult[o];
    for (int k = 0; k < K; ++k) {
      const int32_t acc = ternary_dot_decoded(X + static_cast<size_t>(k) * hidden_dim, wrow, hidden_dim);
      Y[static_cast<size_t>(k) * out_dim + o] = requantize_i32(acc, m, requant_shift);
    }
  }
  std::free(wrow);
  return SPECTRA_OK;
}

int spectra_sparse_ternary_gemv(
    const int8_t* X, size_t x_len,
    const int32_t* active_idx, size_t num_active,
    const uint8_t* W_packed, size_t w_len,
    const int32_t* requant_mult, size_t mult_len,
    int requant_shift, int num_tokens, int hidden_dim, int out_dim,
    int8_t* Y, size_t y_len) {
  const int status = validate_common(
      X, x_len, W_packed, w_len, requant_mult, mult_len,
      requant_shift, out_dim, num_tokens, hidden_dim, Y, y_len);
  if (status != SPECTRA_OK) return status;
  if (num_active > 0 && !active_idx) return SPECTRA_NULL_POINTER;
  for (size_t t = 0; t < num_active; ++t) {
    if (active_idx[t] < 0 || active_idx[t] >= num_tokens) return SPECTRA_ACTIVE_INDEX_OOB;
  }

  const size_t packed_row = packed_row_bytes(hidden_dim);
  // Duplicate active indices are valid by contract: they recompute and overwrite
  // the same output row; no accumulation occurs. Zero active indices is a no-op.
  for (size_t t = 0; t < num_active; ++t) {
    const int token = active_idx[t];
    const int8_t* x = X + static_cast<size_t>(token) * hidden_dim;
    int8_t* y = Y + static_cast<size_t>(token) * out_dim;
    for (int o = 0; o < out_dim; ++o) {
      const uint8_t* w = W_packed + static_cast<size_t>(o) * packed_row;
      const int32_t acc = ternary_dot(x, w, hidden_dim);
      y[o] = requantize_i32(acc, requant_mult[o], requant_shift);
    }
  }
  return SPECTRA_OK;
}

int spectra_fused_ternary_ffn(
    const int8_t* X, size_t x_len,
    const int32_t* active_idx, size_t num_active,
    const uint8_t* W1_packed, size_t w1_len,
    const int32_t* mult1, size_t mult1_len,
    const uint8_t* W2_packed, size_t w2_len,
    const int32_t* mult2, size_t mult2_len,
    int requant_shift, int num_tokens, int hidden_dim, int inter_dim, int out_dim,
    int8_t* Y, size_t y_len) {
  if (inter_dim <= 0 || inter_dim > kMaxDotWidth) return SPECTRA_INVALID_DIMENSION;
  if (hidden_dim <= 0 || hidden_dim > kMaxDotWidth || num_tokens <= 0 || out_dim <= 0) {
    return SPECTRA_INVALID_DIMENSION;
  }
  if (requant_shift < 0 || requant_shift > 62) return SPECTRA_INVALID_SHIFT;
  size_t expected_x = 0, expected_w1 = 0, expected_w2 = 0, expected_y = 0;
  if (!checked_product(static_cast<size_t>(num_tokens), static_cast<size_t>(hidden_dim), &expected_x) ||
      !checked_product(static_cast<size_t>(inter_dim), packed_row_bytes(hidden_dim), &expected_w1) ||
      !checked_product(static_cast<size_t>(out_dim), packed_row_bytes(inter_dim), &expected_w2) ||
      !checked_product(static_cast<size_t>(num_tokens), static_cast<size_t>(out_dim), &expected_y)) {
    return SPECTRA_INVALID_LENGTH;
  }
  if (x_len != expected_x || w1_len != expected_w1 || mult1_len != static_cast<size_t>(inter_dim) ||
      w2_len != expected_w2 || mult2_len != static_cast<size_t>(out_dim) || y_len != expected_y) {
    return SPECTRA_INVALID_LENGTH;
  }
  if (!X || !W1_packed || !mult1 || !W2_packed || !mult2 || !Y) return SPECTRA_NULL_POINTER;
  if (num_active > 0 && !active_idx) return SPECTRA_NULL_POINTER;
  if (!valid_packed_rows(W1_packed, inter_dim, hidden_dim) ||
      !valid_packed_rows(W2_packed, out_dim, inter_dim)) return SPECTRA_INVALID_PACKED_WEIGHT;
  for (int i = 0; i < inter_dim; ++i) if (mult1[i] < 0) return SPECTRA_INVALID_MULTIPLIER;
  for (int i = 0; i < out_dim; ++i) if (mult2[i] < 0) return SPECTRA_INVALID_MULTIPLIER;
  for (size_t t = 0; t < num_active; ++t) {
    if (active_idx[t] < 0 || active_idx[t] >= num_tokens) return SPECTRA_ACTIVE_INDEX_OOB;
  }

  const size_t p1 = packed_row_bytes(hidden_dim);
  const size_t p2 = packed_row_bytes(inter_dim);
  int8_t* tmp = static_cast<int8_t*>(std::malloc(static_cast<size_t>(inter_dim)));
  if (!tmp) return SPECTRA_ALLOCATION_FAILURE;
  for (size_t t = 0; t < num_active; ++t) {
    const int token = active_idx[t];
    const int8_t* x = X + static_cast<size_t>(token) * hidden_dim;
    for (int o = 0; o < inter_dim; ++o) {
      const int32_t acc = ternary_dot(x, W1_packed + static_cast<size_t>(o) * p1, hidden_dim);
      const int8_t v = requantize_i32(acc, mult1[o], requant_shift);
      tmp[o] = v > 0 ? v : 0;
    }
    int8_t* y = Y + static_cast<size_t>(token) * out_dim;
    for (int o = 0; o < out_dim; ++o) {
      const int32_t acc = ternary_dot(tmp, W2_packed + static_cast<size_t>(o) * p2, inter_dim);
      y[o] = requantize_i32(acc, mult2[o], requant_shift);
    }
  }
  std::free(tmp);
  return SPECTRA_OK;
}

}  // extern "C"
