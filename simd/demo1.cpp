#include <emmintrin.h>
#include <immintrin.h>
#include <mmintrin>
#pragma GCC target("avx2")
#pragma GCC optimize("O3")

#include <x86intrin.h>
#include <bits/stdc++.h>
#include <iostream>
using namespace std;

__m128i bitwiseNot( __m128i x )
{
    const __m128i zero = _mm_setzero_si128();
    const __m128i one = _mm_cmpeq_epi32( zero, zero );
    return _mm_xor_si128( x, one );
}

int main() {
    double a[100], b[100], c[100];
    // iterate in blocks of 4,
    // because that's how many doubles can fit into a 256-bit register
    for (int i = 0; i < 100; i += 4) {
        // load two 256-bit segments into registers
        __m256d x = _mm256_loadu_pd(&a[i]);
        __m256d y = _mm256_loadu_pd(&b[i]);

        // add 4+4 64-bit numbers together
        __m256d z = _mm256_add_pd(x, y);

        // write the 256-bit result into memory, starting with c[i]
        _mm256_storeu_pd(&c[i], z);
    }

    for (int i = 0; i < 100; i += 2) {
        __m128d x = _mm_loadu_pd(&a[i]);
        __m128d y = _mm_loadu_pd(&b[i]);
        __m128d z = _mm_add_pd(x, y);

        // single-lane load  ss sd
        __m128d sx = _mm_load_sd(&a[i]);
        __m128d sy = _mm_load_sd(&b[i]);
        __m128d sz = _mm_add_sd(sx, sy);

        _mm_store_pd(&c[i], z);
    }

    __m128 a128 = _mm_setzero_ps();
    __m128d a128d = _mm_setzero_pd();
    __m128i a128i = _mm_setzero_si128();

    __m256 a256 = _mm256_setzero_ps();
    __m256d a256d = _mm256_setzero_pd();
    __m256i a256i = _mm256_setzero_si256();

    return 0;
}