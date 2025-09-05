// #pragma GCC target("avx2")
// #pragma GCC optimize("O3")

#include <iostream>
#include <chrono>

const int n = 1e5;
int a[n], s = 0;

int main() {
    const auto start{std::chrono::steady_clock::now()};
    
    for (int t = 0; t < 100000; t++)
        for (int i = 0; i < n; i++)
            s += a[i];
    
    const auto finish{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{finish - start};
    std::cout << " cost time: " << elapsed_seconds.count() << "\n";
    return 0;
}

// root@f6520cc4c218:~/ws/simd_ws# g++ demo0.cpp -o run
//  cost time: 42.1015
// root@f6520cc4c218:~/ws/simd_ws# g++ -Ofast demo0.cpp -o run
//  cost time: 0.481249
// root@f6520cc4c218:~/ws/simd_ws# g++ -Ofast demo0.cpp -o run
//  cost time: 0.248053
