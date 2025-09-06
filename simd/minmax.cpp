#include <immintrin.h>
#include <iostream>
#include <cmath>
#include <iomanip>

#pragma GCC target("avx")
#pragma GCC target("avx2")
#pragma GCC target("fma")
#pragma GCC optimize("O3")

// 类型定义
using Packet4f = __m128;

// 基础操作函数
Packet4f pcmp_eq(const Packet4f& a, const Packet4f& b) {
    return _mm_cmpeq_ps(a, b);
}

Packet4f pmin(const Packet4f& a, const Packet4f& b) {
    return _mm_min_ps(a, b);
}

Packet4f pmax(const Packet4f& a, const Packet4f& b) {
    return _mm_max_ps(a, b);
}

template <typename Packet>
inline Packet pselect(const Packet& mask, const Packet& a, const Packet& b) {
    // mask ? a : b
    return _mm_or_ps(_mm_and_ps(mask, a), _mm_andnot_ps(mask, b));
}

// ============================================================================
// NaN传播策略实现
// ============================================================================

// 策略1：传播数字（忽略NaN）
template <typename Packet, typename Op>
inline Packet pminmax_propagate_numbers(const Packet& a, const Packet& b, Op op) {
    // 检测a中的NaN
    Packet not_nan_mask_a = pcmp_eq(a, a);
    
    // 执行操作
    Packet m = op(a, b);
    
    // 如果a不是NaN使用m，否则使用b
    return pselect<Packet>(not_nan_mask_a, m, b);
}

// 策略2：传播NaN
template <typename Packet, typename Op>
inline Packet pminmax_propagate_nan(const Packet& a, const Packet& b, Op op) {
    // 检测a中的NaN
    Packet not_nan_mask_a = pcmp_eq(a, a);
    
    // 执行操作（注意参数顺序）
    Packet m = op(b, a);
    
    // 如果a不是NaN使用m，否则使用a（保持NaN）
    return pselect<Packet>(not_nan_mask_a, m, a);
}

// ============================================================================
// 具体的min/max操作
// ============================================================================

// Min操作 - 传播数字
inline Packet4f pmin_propagate_numbers(const Packet4f& a, const Packet4f& b) {
    return pminmax_propagate_numbers(a, b, [](const Packet4f& x, const Packet4f& y) {
        return pmin(x, y);
    });
}

// Min操作 - 传播NaN
inline Packet4f pmin_propagate_nan(const Packet4f& a, const Packet4f& b) {
    return pminmax_propagate_nan(a, b, [](const Packet4f& x, const Packet4f& y) {
        return pmin(x, y);
    });
}

// Max操作 - 传播数字
inline Packet4f pmax_propagate_numbers(const Packet4f& a, const Packet4f& b) {
    return pminmax_propagate_numbers(a, b, [](const Packet4f& x, const Packet4f& y) {
        return pmax(x, y);
    });
}

// Max操作 - 传播NaN
inline Packet4f pmax_propagate_nan(const Packet4f& a, const Packet4f& b) {
    return pminmax_propagate_nan(a, b, [](const Packet4f& x, const Packet4f& y) {
        return pmax(x, y);
    });
}

// ============================================================================
// 测试和演示
// ============================================================================

void print_packet(const char* name, const Packet4f& p) {
    float values[4];
    _mm_store_ps(values, p);
    
    std::cout << name << ": [";
    for (int i = 3; i >= 0; i--) {  // SSE存储是反序的
        if (std::isnan(values[i])) {
            std::cout << "NaN";
        } else {
            std::cout << std::fixed << std::setprecision(1) << values[i];
        }
        if (i > 0) std::cout << ", ";
    }
    std::cout << "]\n";
}

void demonstrate_nan_propagation() {
    std::cout << "=== NaN传播策略演示 ===\n\n";
    
    // 创建测试数据
    Packet4f a = _mm_set_ps(NAN, 2.0f, 5.0f, 1.0f);     // [1.0, 5.0, 2.0, NaN]
    Packet4f b = _mm_set_ps(3.0f, 1.0f, 6.0f, 4.0f);     // [4.0, 6.0, 1.0, 3.0]
    
    print_packet("输入 a", a);
    print_packet("输入 b", b);
    std::cout << "\n";
    
    // 1. 标准SSE行为
    Packet4f sse_min = pmin(a, b);
    Packet4f sse_max = pmax(a, b);
    
    std::cout << "标准SSE行为:\n";
    print_packet("pmin(a, b)", sse_min);
    print_packet("pmax(a, b)", sse_max);
    std::cout << "\n";
    
    // 2. 传播数字策略
    Packet4f min_numbers = pmin_propagate_numbers(a, b);
    Packet4f max_numbers = pmax_propagate_numbers(a, b);
    
    std::cout << "传播数字策略 (忽略NaN):\n";
    print_packet("min_propagate_numbers", min_numbers);
    print_packet("max_propagate_numbers", max_numbers);
    std::cout << "\n";
    
    // 3. 传播NaN策略
    Packet4f min_nan = pmin_propagate_nan(a, b);
    Packet4f max_nan = pmax_propagate_nan(a, b);
    
    std::cout << "传播NaN策略:\n";
    print_packet("min_propagate_nan", min_nan);
    print_packet("max_propagate_nan", max_nan);
    std::cout << "\n";
    
    // 4. 边界情况测试
    std::cout << "=== 边界情况测试 ===\n";
    
    // 两个都是NaN
    Packet4f nan_a = _mm_set_ps(NAN, NAN, 1.0f, 2.0f);
    Packet4f nan_b = _mm_set_ps(NAN, 3.0f, NAN, 4.0f);
    
    print_packet("nan_a", nan_a);
    print_packet("nan_b", nan_b);
    
    Packet4f result_numbers = pmin_propagate_numbers(nan_a, nan_b);
    Packet4f result_nan = pmin_propagate_nan(nan_a, nan_b);
    
    std::cout << "\n结果:\n";
    print_packet("propagate_numbers", result_numbers);
    print_packet("propagate_nan", result_nan);
}

// ============================================================================
// 性能对比
// ============================================================================

#include <chrono>
#include <vector>
#include <random>

void performance_comparison() {
    std::cout << "\n=== 性能对比 ===\n";
    
    const size_t num_operations = 1000000;
    const size_t num_packets = num_operations / 4;  // 每个packet包含4个float
    
    // 生成测试数据
    std::vector<Packet4f> data_a, data_b;
    data_a.reserve(num_packets);
    data_b.reserve(num_packets);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-100.0f, 100.0f);
    std::uniform_int_distribution<int> nan_dis(0, 20);  // 5%概率为NaN
    
    for (size_t i = 0; i < num_packets; ++i) {
        float values_a[4], values_b[4];
        for (int j = 0; j < 4; ++j) {
            values_a[j] = (nan_dis(gen) == 0) ? NAN : dis(gen);
            values_b[j] = (nan_dis(gen) == 0) ? NAN : dis(gen);
        }
        data_a.push_back(_mm_set_ps(values_a[3], values_a[2], values_a[1], values_a[0]));
        data_b.push_back(_mm_set_ps(values_b[3], values_b[2], values_b[1], values_b[0]));
    }
    
    volatile Packet4f sink;  // 防止优化器消除计算
    
    // 测试标准min
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_packets; ++i) {
        sink = pmin(data_a[i], data_b[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto std_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 测试传播数字
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_packets; ++i) {
        sink = pmin_propagate_numbers(data_a[i], data_b[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    auto numbers_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 测试传播NaN
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_packets; ++i) {
        sink = pmin_propagate_nan(data_a[i], data_b[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    auto nan_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "操作数量: " << num_operations << "\n";
    std::cout << "标准 pmin:           " << std_time.count() << " μs\n";
    std::cout << "传播数字版本:        " << numbers_time.count() << " μs (+" 
              << std::fixed << std::setprecision(1) 
              << ((double)numbers_time.count() / std_time.count() - 1.0) * 100 << "%)\n";
    std::cout << "传播NaN版本:         " << nan_time.count() << " μs (+"
              << ((double)nan_time.count() / std_time.count() - 1.0) * 100 << "%)\n";
}

int main() {
    std::cout << "Eigen NaN传播策略分析\n";
    std::cout << "====================\n\n";
    
    demonstrate_nan_propagation();
    performance_comparison();
    
    return 0;
}