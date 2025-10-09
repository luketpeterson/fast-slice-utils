# fast-slice-utils

Highly optimized slice utilities using SIMD instructions when available

## Overview

`fast-slice-utils` provides optimized implementations of a couple of slice operations using platform-specific SIMD instructions (AVX2, NEON, etc.).

The library is `no_std` compatible.

## Functions

- **`find_prefix_overlap`**: Returns the number of matching bytes at the start of two slices
- **`starts_with`**: A faster replacement for the standard library's [`slice::starts_with`](https://doc.rust-lang.org/std/primitive.slice.html#method.starts_with)

## Platform Support

- **x86_64 with AVX2**: Uses 256-bit AVX2 instructions
- **ARM with NEON**: Uses 128-bit NEON instructions
- **Portable SIMD**: Uses portable SIMD. (requires `nightly` feature)

## Safety (miri issue)

Currently some of the implementations may over-read the provided slices, causing miri to become upset.  The code ensures that it never reads across a page boundary.  If a fix can be found (that doesn't hurt performance too badly) we will gladly integrate it.

## Contributing

Any contribution is welcome including bug fixes, optimizations, additional functions, and support for additional platforms.
