#![doc = include_str!("../README.md")]
#![no_std]
#![cfg_attr(feature = "nightly", allow(internal_features), feature(core_intrinsics))]
#![cfg_attr(feature = "nightly", feature(portable_simd))]

// =======================================================================================
// Various implementations of `find_prefix_overlap`
// =======================================================================================

#[cfg(not(feature = "nightly"))]
#[allow(unused)]
pub(crate) use core::convert::{identity as likely, identity as unlikely};
#[cfg(feature = "nightly")]
#[allow(unused)]
pub(crate) use core::intrinsics::{likely, unlikely};

// GOAT!  UGH!  It turned out scalar paths aren't enough faster to justify having them
// Probably on account of the extra branching causing misprediction
// This code should be deleted eventually, but maybe keep it for a while while we discuss
//
// /// Returns the number of characters shared between two slices
// #[inline]
// pub fn find_prefix_overlap(a: &[u8], b: &[u8]) -> usize {
//     let len = a.len().min(b.len());

//     match len {
//         0 => 0,
//         1 => (unsafe{ a.get_unchecked(0) == b.get_unchecked(0) } as usize),
//         2 => { 
//             let a_word = unsafe{ core::ptr::read_unaligned(a.as_ptr() as *const u16) };
//             let b_word = unsafe{ core::ptr::read_unaligned(b.as_ptr() as *const u16) };
//             let cmp = !(a_word ^ b_word); // equal bytes will be 0xFF
//             let cnt = cmp.trailing_ones();
//             cnt as usize / 8
//         },
//         3 | 4 | 5 | 6 | 7 | 8 => {
//             //GOAT, we need to do a check to make sure we don't over-read a page
//             let a_word = unsafe{ core::ptr::read_unaligned(a.as_ptr() as *const u64) };
//             let b_word = unsafe{ core::ptr::read_unaligned(b.as_ptr() as *const u64) };
//             let cmp = !(a_word ^ b_word); // equal bytes will be 0xFF
//             let cnt = cmp.trailing_ones();
//             let result = cnt as usize / 8;
//             result.min(len)
//         },
//         _ => count_shared_neon(a, b),
//     }
// }

// GOAT!  AGH!! Even this is much slower, even on the zipfian distribution where 70% of the pairs have 0 overlap!!!
//
// /// Returns the number of characters shared between two slices
// #[inline]
// pub fn find_prefix_overlap(a: &[u8], b: &[u8]) -> usize {
//     if a.len() != 0 && b.len() != 0 && unsafe{ a.get_unchecked(0) == b.get_unchecked(0) } {
//         count_shared_neon(a, b)
//     } else {
//         0
//     }
// }

const PAGE_SIZE: usize = 4096;

#[inline(always)]
unsafe fn same_page<const VECTOR_SIZE: usize>(slice: &[u8]) -> bool {
    let address = slice.as_ptr() as usize;
    // Mask to keep only the last 12 bits
    let offset_within_page = address & (PAGE_SIZE - 1);
    // Check if the 16/32/64th byte from the current offset exceeds the page boundary
    offset_within_page < PAGE_SIZE - VECTOR_SIZE
}

/// A simple reference implementation of `find_prefix_overlap` with no fanciness
fn count_shared_reference(p: &[u8], q: &[u8]) -> usize {
    p.iter().zip(q)
        .take_while(|(x, y)| x == y).count()
}

#[cold]
fn count_shared_cold(a: &[u8], b: &[u8]) -> usize {
    count_shared_reference(a, b)
}

#[cfg(target_feature = "avx512f")]
#[inline(always)]
fn count_shared_avx512(p: &[u8], q: &[u8]) -> usize {
    use core::arch::x86_64::*;
    unsafe {
        let pl = p.len();
        let ql = q.len();
        let max_shared = pl.min(ql);
        if unlikely(max_shared == 0) { return 0 }
        let m = (!(0u64 as __mmask64)) >> (64 - max_shared.min(64));
        let pv = _mm512_mask_loadu_epi8(_mm512_setzero_si512(), m, p.as_ptr() as _);
        let qv = _mm512_mask_loadu_epi8(_mm512_setzero_si512(), m, q.as_ptr() as _);
        let ne = !_mm512_cmpeq_epi8_mask(pv, qv);
        let count = _tzcnt_u64(ne);
        if count != 64 || max_shared < 65 {
            (count as usize).min(max_shared)
        } else {
            let new_len = max_shared-64;
            64 + count_shared_avx512(core::slice::from_raw_parts(p.as_ptr().add(64), new_len), core::slice::from_raw_parts(q.as_ptr().add(64), new_len))
        }
    }
}

#[cfg(all(target_feature="avx2", not(miri)))]
#[inline(always)]
fn count_shared_avx2(p: &[u8], q: &[u8]) -> usize {
    use core::arch::x86_64::*;
    unsafe {
        let pl = p.len();
        let ql = q.len();
        let max_shared = pl.min(ql);
        if unlikely(max_shared == 0) { return 0 }
        if likely(same_page::<32>(p) && same_page::<32>(q)) {
            let pv = _mm256_loadu_si256(p.as_ptr() as _);
            let qv = _mm256_loadu_si256(q.as_ptr() as _);
            let ev = _mm256_cmpeq_epi8(pv, qv);
            let ne = !(_mm256_movemask_epi8(ev) as u32);
            let count = _tzcnt_u32(ne);
            if count != 32 || max_shared < 33 {
                (count as usize).min(max_shared)
            } else {
                let new_len = max_shared-32;
                32 + count_shared_avx2(core::slice::from_raw_parts(p.as_ptr().add(32), new_len), core::slice::from_raw_parts(q.as_ptr().add(32), new_len))
            }
        } else {
            count_shared_cold(p, q)
        }
    }
}

#[cfg(all(not(feature = "nightly"), target_arch = "aarch64", target_feature = "neon", not(miri)))]
#[inline(always)]
fn count_shared_neon(p: &[u8], q: &[u8]) -> usize {
    use core::arch::aarch64::*;
    unsafe {
        let pl = p.len();
        let ql = q.len();
        let max_shared = pl.min(ql);
        if unlikely(max_shared == 0) { return 0 }

        if same_page::<16>(p) && same_page::<16>(q) {
            let pv = vld1q_u8(p.as_ptr());
            let qv = vld1q_u8(q.as_ptr());
            let eq = vceqq_u8(pv, qv);

            //UGH! There must be a better way to do this...
            // let neg = vmvnq_u8(eq);
            // let lo: u64 = vgetq_lane_u64(core::mem::transmute(neg), 0);
            // let hi: u64 = vgetq_lane_u64(core::mem::transmute(neg), 1);
            // let count = if lo != 0 {
            //     lo.trailing_zeros()
            // } else {
            //     64 + hi.trailing_zeros()
            // } / 8;

            //UGH! This code is actually a bit faster than the commented out code above.
            // I'm sure I'm just not familiar enough with the neon ISA
            let mut bytes = [core::mem::MaybeUninit::<u8>::uninit(); 16];
            vst1q_u8(bytes.as_mut_ptr().cast(), eq);
            let scalar128 = u128::from_le_bytes(core::mem::transmute(bytes));
            let count = scalar128.trailing_ones() / 8;

            if count != 16 || max_shared < 17 {
                (count as usize).min(max_shared)
            } else {
                let new_len = max_shared-16;
                16 + count_shared_neon(core::slice::from_raw_parts(p.as_ptr().add(16), new_len), core::slice::from_raw_parts(q.as_ptr().add(16), new_len))
            }
        } else {
            return count_shared_cold(p, q);
        }
    }
}

#[cfg(all(feature = "nightly", not(miri)))]
#[inline(always)]
fn count_shared_simd(p: &[u8], q: &[u8]) -> usize {
    use core::simd::{u8x32, cmp::SimdPartialEq};
    unsafe {
        let pl = p.len();
        let ql = q.len();
        let max_shared = pl.min(ql);
        if unlikely(max_shared == 0) { return 0 }
        if same_page::<32>(p) && same_page::<32>(q) {
            let mut p_array = [core::mem::MaybeUninit::<u8>::uninit(); 32];
            core::ptr::copy_nonoverlapping(p.as_ptr().cast(), (&mut p_array).as_mut_ptr(), 32);
            let pv = u8x32::from_array(core::mem::transmute(p_array));
            let mut q_array = [core::mem::MaybeUninit::<u8>::uninit(); 32];
            core::ptr::copy_nonoverlapping(q.as_ptr().cast(), (&mut q_array).as_mut_ptr(), 32);
            let qv = u8x32::from_array(core::mem::transmute(q_array));
            let ev = pv.simd_eq(qv);
            let mask = ev.to_bitmask();
            let count = mask.trailing_ones();
            if count != 32 || max_shared < 33 {
                (count as usize).min(max_shared)
            } else {
                let new_len = max_shared-32;
                32 + count_shared_simd(core::slice::from_raw_parts(p.as_ptr().add(32), new_len), core::slice::from_raw_parts(q.as_ptr().add(32), new_len))
            }
        } else {
            return count_shared_cold(p, q);
        }
    }
}

/// Returns the number of initial characters shared between two slices
///
/// The fastest (as measured by us) implementation is exported based on the platform and features.
///
/// - **AVX512**: AVX512 intrinsics (x86_64, requires nightly)
/// - **AVX2**: AVX2 intrinsics (x86_64)
/// - **NEON**: NEON intrinsics (aarch64)
/// - **Portable SIMD**: Portable SIMD (requires nightly)
/// - **Reference**: Reference scalar implementation
///
/// | AVX-512 | AVX2 | NEON | nightly | miri | Implementation    |
/// |---------|------|------|---------|------|-------------------|
/// | ✓       | -    | ✗    | -       | ✗    | **AVX512**        |
/// | -       | ✓    | ✗    | -       | ✗    | **AVX2**          |
/// | ✗       | ✗    | ✓    | ✗       | ✗    | **NEON**          |
/// | ✗       | ✗    | ✓    | ✓       | ✗    | **Portable SIMD** |
/// | -       | -    | -    | -       | ✓    | **Reference**     |
///
#[inline]
pub fn find_prefix_overlap(a: &[u8], b: &[u8]) -> usize {
    #[cfg(all(target_feature="avx512f", not(miri)))]
    {
        count_shared_avx512(a, b)
    }
    #[cfg(all(target_feature="avx2", not(target_feature="avx512f"), not(miri)))]
    {
        count_shared_avx2(a, b)
    }
    #[cfg(all(not(feature = "nightly"), target_arch = "aarch64", target_feature = "neon", not(miri)))]
    {
        count_shared_neon(a, b)
    }
    #[cfg(all(feature = "nightly", target_arch = "aarch64", target_feature = "neon", not(miri)))]
    {
        count_shared_simd(a, b)
    }
    #[cfg(any(all(not(target_feature="avx2"), not(target_feature="neon")), miri))]
    {
        count_shared_reference(a, b)
    }
}

#[test]
fn find_prefix_overlap_test() {
    let tests = [
        ("12345", "67890", 0),
        ("", "12300", 0),
        ("12345", "", 0),
        ("12345", "12300", 3),
        ("123", "123000000", 3),
        ("123456789012345678901234567890xxxx", "123456789012345678901234567890yy", 30),
        ("123456789012345678901234567890123456789012345678901234567890xxxx", "123456789012345678901234567890123456789012345678901234567890yy", 60),
        ("1234567890123456xxxx", "1234567890123456yyyyyyy", 16),
        ("123456789012345xxxx", "123456789012345yyyyyyy", 15),
        ("12345678901234567xxxx", "12345678901234567yyyyyyy", 17),
        ("1234567890123456789012345678901xxxx", "1234567890123456789012345678901yy", 31),
        ("12345678901234567890123456789012xxxx", "12345678901234567890123456789012yy", 32),
        ("123456789012345678901234567890123xxxx", "123456789012345678901234567890123yy", 33),
        ("123456789012345678901234567890123456789012345678901234567890123xxxx", "123456789012345678901234567890123456789012345678901234567890123yy", 63),
        ("1234567890123456789012345678901234567890123456789012345678901234xxxx", "1234567890123456789012345678901234567890123456789012345678901234yy", 64),
        ("12345678901234567890123456789012345678901234567890123456789012345xxxx", "12345678901234567890123456789012345678901234567890123456789012345yy", 65),
    ];

    for test in tests {
        let overlap = find_prefix_overlap(test.0.as_bytes(), test.1.as_bytes());
        assert_eq!(overlap, test.2);
    }
}

/// A faster replacement for the stdlib version of [`starts_with`](slice::starts_with)
#[inline(always)]
pub fn starts_with(x: &[u8], y: &[u8]) -> bool {
    if y.len() == 0 { return true }
    if x.len() == 0 { return false }
    if y.len() > x.len() { return false }
    find_prefix_overlap(x, y) == y.len()
}
