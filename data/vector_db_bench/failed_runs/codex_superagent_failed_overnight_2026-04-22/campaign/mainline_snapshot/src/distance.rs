use half::f16;

pub const DIM: usize = 128;

#[inline(always)]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    l2_squared(a, b) as f64
}

#[inline(always)]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), DIM);
    debug_assert_eq!(b.len(), DIM);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { l2_squared_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { l2_squared_avx2(a, b) };
        }
    }

    l2_squared_scalar(a, b)
}

#[inline(always)]
pub fn nearest_centroid(query: &[f32], centroids_flat: &[f32], count: usize) -> usize {
    debug_assert_eq!(query.len(), DIM);
    debug_assert!(centroids_flat.len() >= count * DIM);

    let mut best_idx = 0usize;
    let mut best_dist = f32::MAX;

    for idx in 0..count {
        let start = idx * DIM;
        let dist = l2_squared(query, &centroids_flat[start..start + DIM]);
        if dist < best_dist {
            best_dist = dist;
            best_idx = idx;
        }
    }

    best_idx
}

#[inline(always)]
pub fn l2_distance_batch(
    query: &[f32],
    vectors_flat: &[f32],
    n_vectors: usize,
    distances: &mut [f32],
) {
    debug_assert_eq!(query.len(), DIM);
    debug_assert!(vectors_flat.len() >= n_vectors * DIM);
    debug_assert!(distances.len() >= n_vectors);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { l2_distance_batch_avx512(query, vectors_flat, n_vectors, distances) };
            return;
        }
        if is_x86_feature_detected!("avx2") {
            unsafe { l2_distance_batch_avx2(query, vectors_flat, n_vectors, distances) };
            return;
        }
    }

    for idx in 0..n_vectors {
        let start = idx * DIM;
        distances[idx] = l2_squared_scalar(query, &vectors_flat[start..start + DIM]);
    }
}

#[inline(always)]
pub fn l2_distance_batch_f16(
    query: &[f32],
    vectors_flat: &[f16],
    n_vectors: usize,
    distances: &mut [f32],
) {
    debug_assert_eq!(query.len(), DIM);
    debug_assert!(vectors_flat.len() >= n_vectors * DIM);
    debug_assert!(distances.len() >= n_vectors);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") && is_x86_feature_detected!("f16c") {
            unsafe { l2_distance_batch_f16_f16c(query, vectors_flat, n_vectors, distances) };
            return;
        }
    }

    for idx in 0..n_vectors {
        let start = idx * DIM;
        distances[idx] = l2_squared_f16(query, &vectors_flat[start..start + DIM]);
    }
}

#[inline(always)]
pub fn l2_squared_f16(query: &[f32], vector: &[f16]) -> f32 {
    debug_assert_eq!(query.len(), DIM);
    debug_assert_eq!(vector.len(), DIM);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") && is_x86_feature_detected!("f16c") {
            return unsafe { l2_squared_f16_f16c(query, vector) };
        }
    }

    let mut sum = 0.0f32;
    for idx in 0..DIM {
        let delta = query[idx] - vector[idx].to_f32();
        sum += delta * delta;
    }
    sum
}

pub fn l2_squared_u8(query: &[u8; DIM], vector: &[u8]) -> u32 {
    debug_assert_eq!(vector.len(), DIM);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { l2_squared_u8_avx2(query, vector) };
        }
    }

    l2_squared_u8_scalar(query, vector)
}

#[inline(always)]
pub fn l2_squared_u8_slice(query: &[u8], vector: &[u8]) -> u32 {
    debug_assert_eq!(query.len(), vector.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { l2_squared_u8_slice_avx2(query, vector) };
        }
    }

    l2_squared_u8_slice_scalar(query, vector)
}

#[inline(always)]
pub fn l2_squared_u8_slice_with_upper_bound(query: &[u8], vector: &[u8], upper_bound: u32) -> u32 {
    debug_assert_eq!(query.len(), vector.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe {
                l2_squared_u8_slice_with_upper_bound_avx2(query, vector, upper_bound)
            };
        }
    }

    let mut sum = 0u32;
    for idx in 0..query.len() {
        let delta = query[idx] as i32 - vector[idx] as i32;
        sum += (delta * delta) as u32;
        if sum >= upper_bound {
            return sum;
        }
    }
    sum
}

#[inline(always)]
pub fn l2_distance_batch_u8(
    query: &[u8; DIM],
    vectors_flat: &[u8],
    n_vectors: usize,
    distances: &mut [u32],
) {
    debug_assert!(vectors_flat.len() >= n_vectors * DIM);
    debug_assert!(distances.len() >= n_vectors);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { l2_distance_batch_u8_avx2(query, vectors_flat, n_vectors, distances) };
            return;
        }
    }

    for idx in 0..n_vectors {
        let start = idx * DIM;
        distances[idx] = l2_squared_u8(query, &vectors_flat[start..start + DIM]);
    }
}

#[inline(always)]
pub fn l2_distance_batch_u8_slice(
    query: &[u8],
    vectors_flat: &[u8],
    dims: usize,
    n_vectors: usize,
    distances: &mut [u32],
) {
    debug_assert_eq!(query.len(), dims);
    debug_assert!(vectors_flat.len() >= n_vectors * dims);
    debug_assert!(distances.len() >= n_vectors);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                l2_distance_batch_u8_slice_avx2(query, vectors_flat, dims, n_vectors, distances)
            };
            return;
        }
    }

    for idx in 0..n_vectors {
        let start = idx * dims;
        distances[idx] = l2_squared_u8_slice_scalar(query, &vectors_flat[start..start + dims]);
    }
}

#[inline(always)]
fn l2_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;

    for idx in 0..DIM {
        let delta = a[idx] - b[idx];
        sum += delta * delta;
    }

    sum
}

#[inline(always)]
fn l2_squared_u8_scalar(query: &[u8; DIM], vector: &[u8]) -> u32 {
    let mut sum = 0u32;

    for idx in 0..DIM {
        let delta = query[idx] as i32 - vector[idx] as i32;
        sum += (delta * delta) as u32;
    }

    sum
}

#[inline(always)]
fn l2_squared_u8_slice_scalar(query: &[u8], vector: &[u8]) -> u32 {
    let mut sum = 0u32;

    for idx in 0..query.len() {
        let delta = query[idx] as i32 - vector[idx] as i32;
        sum += (delta * delta) as u32;
    }

    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn l2_squared_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let d0 = _mm512_sub_ps(_mm512_loadu_ps(a_ptr), _mm512_loadu_ps(b_ptr));
    let mut sum = _mm512_mul_ps(d0, d0);

    let d1 = _mm512_sub_ps(
        _mm512_loadu_ps(a_ptr.add(16)),
        _mm512_loadu_ps(b_ptr.add(16)),
    );
    sum = _mm512_fmadd_ps(d1, d1, sum);

    let d2 = _mm512_sub_ps(
        _mm512_loadu_ps(a_ptr.add(32)),
        _mm512_loadu_ps(b_ptr.add(32)),
    );
    sum = _mm512_fmadd_ps(d2, d2, sum);

    let d3 = _mm512_sub_ps(
        _mm512_loadu_ps(a_ptr.add(48)),
        _mm512_loadu_ps(b_ptr.add(48)),
    );
    sum = _mm512_fmadd_ps(d3, d3, sum);

    let d4 = _mm512_sub_ps(
        _mm512_loadu_ps(a_ptr.add(64)),
        _mm512_loadu_ps(b_ptr.add(64)),
    );
    sum = _mm512_fmadd_ps(d4, d4, sum);

    let d5 = _mm512_sub_ps(
        _mm512_loadu_ps(a_ptr.add(80)),
        _mm512_loadu_ps(b_ptr.add(80)),
    );
    sum = _mm512_fmadd_ps(d5, d5, sum);

    let d6 = _mm512_sub_ps(
        _mm512_loadu_ps(a_ptr.add(96)),
        _mm512_loadu_ps(b_ptr.add(96)),
    );
    sum = _mm512_fmadd_ps(d6, d6, sum);

    let d7 = _mm512_sub_ps(
        _mm512_loadu_ps(a_ptr.add(112)),
        _mm512_loadu_ps(b_ptr.add(112)),
    );
    sum = _mm512_fmadd_ps(d7, d7, sum);

    _mm512_reduce_add_ps(sum)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn l2_distance_batch_avx512(
    query: &[f32],
    vectors_flat: &[f32],
    n_vectors: usize,
    distances: &mut [f32],
) {
    use std::arch::x86_64::*;

    let q_ptr = query.as_ptr();
    let q0 = _mm512_loadu_ps(q_ptr);
    let q1 = _mm512_loadu_ps(q_ptr.add(16));
    let q2 = _mm512_loadu_ps(q_ptr.add(32));
    let q3 = _mm512_loadu_ps(q_ptr.add(48));
    let q4 = _mm512_loadu_ps(q_ptr.add(64));
    let q5 = _mm512_loadu_ps(q_ptr.add(80));
    let q6 = _mm512_loadu_ps(q_ptr.add(96));
    let q7 = _mm512_loadu_ps(q_ptr.add(112));

    let v_ptr = vectors_flat.as_ptr();

    for idx in 0..n_vectors {
        let base = v_ptr.add(idx * DIM);

        let d0 = _mm512_sub_ps(q0, _mm512_loadu_ps(base));
        let mut sum = _mm512_mul_ps(d0, d0);

        let d1 = _mm512_sub_ps(q1, _mm512_loadu_ps(base.add(16)));
        sum = _mm512_fmadd_ps(d1, d1, sum);

        let d2 = _mm512_sub_ps(q2, _mm512_loadu_ps(base.add(32)));
        sum = _mm512_fmadd_ps(d2, d2, sum);

        let d3 = _mm512_sub_ps(q3, _mm512_loadu_ps(base.add(48)));
        sum = _mm512_fmadd_ps(d3, d3, sum);

        let d4 = _mm512_sub_ps(q4, _mm512_loadu_ps(base.add(64)));
        sum = _mm512_fmadd_ps(d4, d4, sum);

        let d5 = _mm512_sub_ps(q5, _mm512_loadu_ps(base.add(80)));
        sum = _mm512_fmadd_ps(d5, d5, sum);

        let d6 = _mm512_sub_ps(q6, _mm512_loadu_ps(base.add(96)));
        sum = _mm512_fmadd_ps(d6, d6, sum);

        let d7 = _mm512_sub_ps(q7, _mm512_loadu_ps(base.add(112)));
        sum = _mm512_fmadd_ps(d7, d7, sum);

        distances[idx] = _mm512_reduce_add_ps(sum);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    for offset in (0..DIM).step_by(32) {
        let d0 = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(offset)),
            _mm256_loadu_ps(b_ptr.add(offset)),
        );
        let d1 = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 8)),
            _mm256_loadu_ps(b_ptr.add(offset + 8)),
        );
        let d2 = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 16)),
            _mm256_loadu_ps(b_ptr.add(offset + 16)),
        );
        let d3 = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(offset + 24)),
            _mm256_loadu_ps(b_ptr.add(offset + 24)),
        );
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(d0, d0));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(d1, d1));
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(d2, d2));
        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(d3, d3));
    }

    horizontal_sum_m256(_mm256_add_ps(
        _mm256_add_ps(sum0, sum1),
        _mm256_add_ps(sum2, sum3),
    ))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_distance_batch_avx2(
    query: &[f32],
    vectors_flat: &[f32],
    n_vectors: usize,
    distances: &mut [f32],
) {
    use std::arch::x86_64::*;

    let q_ptr = query.as_ptr();
    let q0 = _mm256_loadu_ps(q_ptr);
    let q1 = _mm256_loadu_ps(q_ptr.add(8));
    let q2 = _mm256_loadu_ps(q_ptr.add(16));
    let q3 = _mm256_loadu_ps(q_ptr.add(24));
    let q4 = _mm256_loadu_ps(q_ptr.add(32));
    let q5 = _mm256_loadu_ps(q_ptr.add(40));
    let q6 = _mm256_loadu_ps(q_ptr.add(48));
    let q7 = _mm256_loadu_ps(q_ptr.add(56));
    let q8 = _mm256_loadu_ps(q_ptr.add(64));
    let q9 = _mm256_loadu_ps(q_ptr.add(72));
    let q10 = _mm256_loadu_ps(q_ptr.add(80));
    let q11 = _mm256_loadu_ps(q_ptr.add(88));
    let q12 = _mm256_loadu_ps(q_ptr.add(96));
    let q13 = _mm256_loadu_ps(q_ptr.add(104));
    let q14 = _mm256_loadu_ps(q_ptr.add(112));
    let q15 = _mm256_loadu_ps(q_ptr.add(120));

    let v_ptr = vectors_flat.as_ptr();

    for idx in 0..n_vectors {
        let base = v_ptr.add(idx * DIM);

        let d0 = _mm256_sub_ps(q0, _mm256_loadu_ps(base));
        let d1 = _mm256_sub_ps(q1, _mm256_loadu_ps(base.add(8)));
        let d2 = _mm256_sub_ps(q2, _mm256_loadu_ps(base.add(16)));
        let d3 = _mm256_sub_ps(q3, _mm256_loadu_ps(base.add(24)));
        let d4 = _mm256_sub_ps(q4, _mm256_loadu_ps(base.add(32)));
        let d5 = _mm256_sub_ps(q5, _mm256_loadu_ps(base.add(40)));
        let d6 = _mm256_sub_ps(q6, _mm256_loadu_ps(base.add(48)));
        let d7 = _mm256_sub_ps(q7, _mm256_loadu_ps(base.add(56)));
        let d8 = _mm256_sub_ps(q8, _mm256_loadu_ps(base.add(64)));
        let d9 = _mm256_sub_ps(q9, _mm256_loadu_ps(base.add(72)));
        let d10 = _mm256_sub_ps(q10, _mm256_loadu_ps(base.add(80)));
        let d11 = _mm256_sub_ps(q11, _mm256_loadu_ps(base.add(88)));
        let d12 = _mm256_sub_ps(q12, _mm256_loadu_ps(base.add(96)));
        let d13 = _mm256_sub_ps(q13, _mm256_loadu_ps(base.add(104)));
        let d14 = _mm256_sub_ps(q14, _mm256_loadu_ps(base.add(112)));
        let d15 = _mm256_sub_ps(q15, _mm256_loadu_ps(base.add(120)));

        let sum0 = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(d0, d0), _mm256_mul_ps(d1, d1)),
            _mm256_add_ps(_mm256_mul_ps(d2, d2), _mm256_mul_ps(d3, d3)),
        );
        let sum1 = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(d4, d4), _mm256_mul_ps(d5, d5)),
            _mm256_add_ps(_mm256_mul_ps(d6, d6), _mm256_mul_ps(d7, d7)),
        );
        let sum2 = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(d8, d8), _mm256_mul_ps(d9, d9)),
            _mm256_add_ps(_mm256_mul_ps(d10, d10), _mm256_mul_ps(d11, d11)),
        );
        let sum3 = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(d12, d12), _mm256_mul_ps(d13, d13)),
            _mm256_add_ps(_mm256_mul_ps(d14, d14), _mm256_mul_ps(d15, d15)),
        );

        distances[idx] = horizontal_sum_m256(_mm256_add_ps(
            _mm256_add_ps(sum0, sum1),
            _mm256_add_ps(sum2, sum3),
        ));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn horizontal_sum_m256(value: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;

    let lo = _mm256_castps256_ps128(value);
    let hi = _mm256_extractf128_ps(value, 1);
    let sum = _mm_add_ps(lo, hi);
    let sum = _mm_hadd_ps(sum, sum);
    let sum = _mm_hadd_ps(sum, sum);
    _mm_cvtss_f32(sum)
}

#[target_feature(enable = "avx,f16c")]
unsafe fn l2_squared_f16_f16c(query: &[f32], vector: &[f16]) -> f32 {
    use std::arch::x86_64::*;

    let q_ptr = query.as_ptr();
    let v_ptr = vector.as_ptr() as *const u16;
    let mut sum = _mm256_setzero_ps();

    for offset in (0..DIM).step_by(8) {
        let q = _mm256_loadu_ps(q_ptr.add(offset));
        let v = _mm_loadu_si128(v_ptr.add(offset) as *const __m128i);
        let vf = _mm256_cvtph_ps(v);
        let delta = _mm256_sub_ps(q, vf);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(delta, delta));
    }

    let lo = _mm256_castps256_ps128(sum);
    let hi = _mm256_extractf128_ps(sum, 1);
    let sum = _mm_add_ps(lo, hi);
    let sum = _mm_hadd_ps(sum, sum);
    let sum = _mm_hadd_ps(sum, sum);
    _mm_cvtss_f32(sum)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn l2_distance_batch_f16_f16c(
    query: &[f32],
    vectors_flat: &[f16],
    n_vectors: usize,
    distances: &mut [f32],
) {
    use std::arch::x86_64::*;

    let q_ptr = query.as_ptr();
    let q0 = _mm256_loadu_ps(q_ptr);
    let q1 = _mm256_loadu_ps(q_ptr.add(8));
    let q2 = _mm256_loadu_ps(q_ptr.add(16));
    let q3 = _mm256_loadu_ps(q_ptr.add(24));
    let q4 = _mm256_loadu_ps(q_ptr.add(32));
    let q5 = _mm256_loadu_ps(q_ptr.add(40));
    let q6 = _mm256_loadu_ps(q_ptr.add(48));
    let q7 = _mm256_loadu_ps(q_ptr.add(56));
    let q8 = _mm256_loadu_ps(q_ptr.add(64));
    let q9 = _mm256_loadu_ps(q_ptr.add(72));
    let q10 = _mm256_loadu_ps(q_ptr.add(80));
    let q11 = _mm256_loadu_ps(q_ptr.add(88));
    let q12 = _mm256_loadu_ps(q_ptr.add(96));
    let q13 = _mm256_loadu_ps(q_ptr.add(104));
    let q14 = _mm256_loadu_ps(q_ptr.add(112));
    let q15 = _mm256_loadu_ps(q_ptr.add(120));

    let v_ptr = vectors_flat.as_ptr() as *const u16;

    for idx in 0..n_vectors {
        let base = v_ptr.add(idx * DIM);
        let d0 = _mm256_sub_ps(q0, _mm256_cvtph_ps(_mm_loadu_si128(base as *const __m128i)));
        let d1 = _mm256_sub_ps(
            q1,
            _mm256_cvtph_ps(_mm_loadu_si128(base.add(8) as *const __m128i)),
        );
        let d2 = _mm256_sub_ps(
            q2,
            _mm256_cvtph_ps(_mm_loadu_si128(base.add(16) as *const __m128i)),
        );
        let d3 = _mm256_sub_ps(
            q3,
            _mm256_cvtph_ps(_mm_loadu_si128(base.add(24) as *const __m128i)),
        );
        let d4 = _mm256_sub_ps(
            q4,
            _mm256_cvtph_ps(_mm_loadu_si128(base.add(32) as *const __m128i)),
        );
        let d5 = _mm256_sub_ps(
            q5,
            _mm256_cvtph_ps(_mm_loadu_si128(base.add(40) as *const __m128i)),
        );
        let d6 = _mm256_sub_ps(
            q6,
            _mm256_cvtph_ps(_mm_loadu_si128(base.add(48) as *const __m128i)),
        );
        let d7 = _mm256_sub_ps(
            q7,
            _mm256_cvtph_ps(_mm_loadu_si128(base.add(56) as *const __m128i)),
        );
        let d8 = _mm256_sub_ps(
            q8,
            _mm256_cvtph_ps(_mm_loadu_si128(base.add(64) as *const __m128i)),
        );
        let d9 = _mm256_sub_ps(
            q9,
            _mm256_cvtph_ps(_mm_loadu_si128(base.add(72) as *const __m128i)),
        );
        let d10 = _mm256_sub_ps(
            q10,
            _mm256_cvtph_ps(_mm_loadu_si128(base.add(80) as *const __m128i)),
        );
        let d11 = _mm256_sub_ps(
            q11,
            _mm256_cvtph_ps(_mm_loadu_si128(base.add(88) as *const __m128i)),
        );
        let d12 = _mm256_sub_ps(
            q12,
            _mm256_cvtph_ps(_mm_loadu_si128(base.add(96) as *const __m128i)),
        );
        let d13 = _mm256_sub_ps(
            q13,
            _mm256_cvtph_ps(_mm_loadu_si128(base.add(104) as *const __m128i)),
        );
        let d14 = _mm256_sub_ps(
            q14,
            _mm256_cvtph_ps(_mm_loadu_si128(base.add(112) as *const __m128i)),
        );
        let d15 = _mm256_sub_ps(
            q15,
            _mm256_cvtph_ps(_mm_loadu_si128(base.add(120) as *const __m128i)),
        );

        let sum0 = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(d0, d0), _mm256_mul_ps(d1, d1)),
            _mm256_add_ps(_mm256_mul_ps(d2, d2), _mm256_mul_ps(d3, d3)),
        );
        let sum1 = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(d4, d4), _mm256_mul_ps(d5, d5)),
            _mm256_add_ps(_mm256_mul_ps(d6, d6), _mm256_mul_ps(d7, d7)),
        );
        let sum2 = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(d8, d8), _mm256_mul_ps(d9, d9)),
            _mm256_add_ps(_mm256_mul_ps(d10, d10), _mm256_mul_ps(d11, d11)),
        );
        let sum3 = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(d12, d12), _mm256_mul_ps(d13, d13)),
            _mm256_add_ps(_mm256_mul_ps(d14, d14), _mm256_mul_ps(d15, d15)),
        );

        distances[idx] = horizontal_sum_m256(_mm256_add_ps(
            _mm256_add_ps(sum0, sum1),
            _mm256_add_ps(sum2, sum3),
        ));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_squared_u8_avx2(query: &[u8; DIM], vector: &[u8]) -> u32 {
    use std::arch::x86_64::*;

    let q_ptr = query.as_ptr();
    let v_ptr = vector.as_ptr();
    let mut sum = _mm256_setzero_si256();

    for offset in (0..DIM).step_by(16) {
        let q = _mm_loadu_si128(q_ptr.add(offset) as *const __m128i);
        let v = _mm_loadu_si128(v_ptr.add(offset) as *const __m128i);
        let q16 = _mm256_cvtepu8_epi16(q);
        let v16 = _mm256_cvtepu8_epi16(v);
        let delta = _mm256_sub_epi16(q16, v16);
        let squared = _mm256_madd_epi16(delta, delta);
        sum = _mm256_add_epi32(sum, squared);
    }

    let lo = _mm256_castsi256_si128(sum);
    let hi = _mm256_extracti128_si256(sum, 1);
    let sum = _mm_add_epi32(lo, hi);
    let sum = _mm_hadd_epi32(sum, sum);
    let sum = _mm_hadd_epi32(sum, sum);
    _mm_cvtsi128_si32(sum) as u32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_distance_batch_u8_avx2(
    query: &[u8; DIM],
    vectors_flat: &[u8],
    n_vectors: usize,
    distances: &mut [u32],
) {
    use std::arch::x86_64::*;

    let q_ptr = query.as_ptr();
    let q0 = _mm256_cvtepu8_epi16(_mm_loadu_si128(q_ptr as *const __m128i));
    let q1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(q_ptr.add(16) as *const __m128i));
    let q2 = _mm256_cvtepu8_epi16(_mm_loadu_si128(q_ptr.add(32) as *const __m128i));
    let q3 = _mm256_cvtepu8_epi16(_mm_loadu_si128(q_ptr.add(48) as *const __m128i));
    let q4 = _mm256_cvtepu8_epi16(_mm_loadu_si128(q_ptr.add(64) as *const __m128i));
    let q5 = _mm256_cvtepu8_epi16(_mm_loadu_si128(q_ptr.add(80) as *const __m128i));
    let q6 = _mm256_cvtepu8_epi16(_mm_loadu_si128(q_ptr.add(96) as *const __m128i));
    let q7 = _mm256_cvtepu8_epi16(_mm_loadu_si128(q_ptr.add(112) as *const __m128i));

    let v_ptr = vectors_flat.as_ptr();
    for idx in 0..n_vectors {
        let base = v_ptr.add(idx * DIM);
        let mut sum = _mm256_setzero_si256();

        let v0 = _mm256_cvtepu8_epi16(_mm_loadu_si128(base as *const __m128i));
        let v1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(base.add(16) as *const __m128i));
        let v2 = _mm256_cvtepu8_epi16(_mm_loadu_si128(base.add(32) as *const __m128i));
        let v3 = _mm256_cvtepu8_epi16(_mm_loadu_si128(base.add(48) as *const __m128i));
        let v4 = _mm256_cvtepu8_epi16(_mm_loadu_si128(base.add(64) as *const __m128i));
        let v5 = _mm256_cvtepu8_epi16(_mm_loadu_si128(base.add(80) as *const __m128i));
        let v6 = _mm256_cvtepu8_epi16(_mm_loadu_si128(base.add(96) as *const __m128i));
        let v7 = _mm256_cvtepu8_epi16(_mm_loadu_si128(base.add(112) as *const __m128i));

        let d0 = _mm256_sub_epi16(q0, v0);
        let d1 = _mm256_sub_epi16(q1, v1);
        let d2 = _mm256_sub_epi16(q2, v2);
        let d3 = _mm256_sub_epi16(q3, v3);
        let d4 = _mm256_sub_epi16(q4, v4);
        let d5 = _mm256_sub_epi16(q5, v5);
        let d6 = _mm256_sub_epi16(q6, v6);
        let d7 = _mm256_sub_epi16(q7, v7);

        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(d0, d0));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(d1, d1));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(d2, d2));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(d3, d3));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(d4, d4));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(d5, d5));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(d6, d6));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(d7, d7));

        let lo = _mm256_castsi256_si128(sum);
        let hi = _mm256_extracti128_si256(sum, 1);
        let sum = _mm_add_epi32(lo, hi);
        let sum = _mm_hadd_epi32(sum, sum);
        let sum = _mm_hadd_epi32(sum, sum);
        distances[idx] = _mm_cvtsi128_si32(sum) as u32;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_squared_u8_slice_avx2(query: &[u8], vector: &[u8]) -> u32 {
    use std::arch::x86_64::*;

    let mut sum = _mm256_setzero_si256();
    let mut offset = 0usize;
    while offset + 16 <= query.len() {
        let q = _mm_loadu_si128(query.as_ptr().add(offset) as *const __m128i);
        let v = _mm_loadu_si128(vector.as_ptr().add(offset) as *const __m128i);
        let q16 = _mm256_cvtepu8_epi16(q);
        let v16 = _mm256_cvtepu8_epi16(v);
        let delta = _mm256_sub_epi16(q16, v16);
        let squared = _mm256_madd_epi16(delta, delta);
        sum = _mm256_add_epi32(sum, squared);
        offset += 16;
    }

    let lo = _mm256_castsi256_si128(sum);
    let hi = _mm256_extracti128_si256(sum, 1);
    let sum = _mm_add_epi32(lo, hi);
    let sum = _mm_hadd_epi32(sum, sum);
    let sum = _mm_hadd_epi32(sum, sum);
    let mut result = _mm_cvtsi128_si32(sum) as u32;

    while offset < query.len() {
        let delta = query[offset] as i32 - vector[offset] as i32;
        result += (delta * delta) as u32;
        offset += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_distance_batch_u8_slice_avx2(
    query: &[u8],
    vectors_flat: &[u8],
    dims: usize,
    n_vectors: usize,
    distances: &mut [u32],
) {
    use std::arch::x86_64::*;

    debug_assert_eq!(query.len(), dims);
    debug_assert!(dims <= DIM);

    let mut query_blocks = [_mm256_setzero_si256(); DIM / 16];
    let mut block_count = 0usize;
    let mut offset = 0usize;
    while offset + 16 <= dims {
        query_blocks[block_count] =
            _mm256_cvtepu8_epi16(_mm_loadu_si128(query.as_ptr().add(offset) as *const __m128i));
        block_count += 1;
        offset += 16;
    }

    let tail_start = offset;
    let v_ptr = vectors_flat.as_ptr();
    for idx in 0..n_vectors {
        let base = v_ptr.add(idx * dims);
        let mut sum = _mm256_setzero_si256();
        let mut block_offset = 0usize;
        for query_block in query_blocks[..block_count].iter() {
            let vector_block =
                _mm256_cvtepu8_epi16(_mm_loadu_si128(base.add(block_offset) as *const __m128i));
            let delta = _mm256_sub_epi16(*query_block, vector_block);
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(delta, delta));
            block_offset += 16;
        }

        let lo = _mm256_castsi256_si128(sum);
        let hi = _mm256_extracti128_si256(sum, 1);
        let sum = _mm_add_epi32(lo, hi);
        let sum = _mm_hadd_epi32(sum, sum);
        let sum = _mm_hadd_epi32(sum, sum);
        let mut result = _mm_cvtsi128_si32(sum) as u32;

        for tail_idx in tail_start..dims {
            let delta = query[tail_idx] as i32 - *base.add(tail_idx) as i32;
            result += (delta * delta) as u32;
        }
        distances[idx] = result;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_squared_u8_slice_with_upper_bound_avx2(
    query: &[u8],
    vector: &[u8],
    upper_bound: u32,
) -> u32 {
    use std::arch::x86_64::*;

    let mut result = 0u32;
    let mut offset = 0usize;

    while offset + 16 <= query.len() {
        let q = _mm_loadu_si128(query.as_ptr().add(offset) as *const __m128i);
        let v = _mm_loadu_si128(vector.as_ptr().add(offset) as *const __m128i);
        let q16 = _mm256_cvtepu8_epi16(q);
        let v16 = _mm256_cvtepu8_epi16(v);
        let delta = _mm256_sub_epi16(q16, v16);
        let squared = _mm256_madd_epi16(delta, delta);

        let lo = _mm256_castsi256_si128(squared);
        let hi = _mm256_extracti128_si256(squared, 1);
        let block = _mm_add_epi32(lo, hi);
        let block = _mm_hadd_epi32(block, block);
        let block = _mm_hadd_epi32(block, block);
        result += _mm_cvtsi128_si32(block) as u32;
        if result >= upper_bound {
            return result;
        }

        offset += 16;
    }

    while offset < query.len() {
        let delta = query[offset] as i32 - vector[offset] as i32;
        result += (delta * delta) as u32;
        if result >= upper_bound {
            return result;
        }
        offset += 1;
    }

    result
}
