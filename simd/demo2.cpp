#include <cstdio>
#include <xmmintrin.h>
#pragma GCC target("avx2")
#pragma GCC optimize("O3")

#include <x86intrin.h>
#include <bits/stdc++.h>
#include <iostream>
using namespace std;

int main() {
    double a[100], b[100], c[100];
    for (int i = 0; i < 100; i++) {
        a[i] = i;
        b[i] = 0;
        c[i] = 0;
    }

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

    std::cout << "check: \n";
    for (int i = 0; i < 100; i++) {
        if (c[i] != a[i] + b[i]) {
            std::cout << "not match.\n";
        }
    }
    std::cout << "check done.\n";
    
    for (int i = 0; i < 100; i++) {
        a[i] = i;
        b[i] = i + 1;
        c[i] = 0;
    }

    // because that's how many doubles can fit into a 128-bit register
    for (int i = 0; i < 100; i += 2) {
        __m128d x = _mm_load_sd(&a[i]);
        __m128d y = _mm_load_sd(&b[i]);

        __m128d z = _mm_add_sd(x, y);

        _mm_store_sd(&c[i], z);
    }

    std::cout << "check step 2: \n";
    for (int i = 0; i < 100; i += 2) {
        if (c[i] != a[i] + b[i]) {
            std::cout << "not match.\n";
        }
    }
    std::cout << "check step 2 done.\n";

    float fa[100], fb[100], fc[100], fd[100];
    for (int i = 0; i < 100; i++) {
        fa[i] = i;
        fb[i] = i + 1;
        fc[i] = 0;
        fd[i] = 0;
    }

    for (int i = 0; i < 100; i += 4) {
        __m128 x = _mm_load_ps(&fa[i]);
        __m128 y = _mm_load_ps(&fb[i]);
        __m128 z = _mm_add_ss(x, y);
        __m128 lz = _mm_add_ps(x, y);

        // __m128 cmp = _mm_cmpgt_ps(lz, z);
        // float cmp_res[4];
        // _mm_store_ps(&cmp_res[0], cmp);
        // std::cout << "*************************\n";
        // for (int i = 0; i < 4; i++) {
        //     std::cout << " -> " << cmp_res[i] << " \n";
        // }
        // std::cout << "*************************\n";

        _mm_store_ss(&fc[i], z);
        _mm_store_ps(&fd[i], lz);
    }
    
    std::cout << "check step 3: \n";
    for (int i = 0; i < 20; i ++) {
        if (fc[i] != fd[i]) {
            std::cout << "not match.\n";
        }
    }
    std::cout << "check step 3 done.\n";

    return 0;
}