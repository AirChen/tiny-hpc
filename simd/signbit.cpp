#include <immintrin.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <vector>

#pragma GCC target("avx")
#pragma GCC target("avx2")
#pragma GCC target("fma")
#pragma GCC optimize("O3")

using Packet4f = __m128;

// Eigen的psignbit实现
Packet4f psignbit(const Packet4f& a) {
    return _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(a), 31));
}

// 工具函数：打印浮点向量
void print_float_vector(const char* name, const Packet4f& v) {
    float values[4];
    _mm_store_ps(values, v);
    
    std::cout << name << ": [";
    for (int i = 3; i >= 0; i--) {  // SSE存储是反序的
        std::cout << std::setw(8) << std::fixed << std::setprecision(2) << values[i];
        if (i > 0) std::cout << ", ";
    }
    std::cout << "]\n";
}

// 工具函数：打印位模式
void print_bit_pattern(const char* name, const Packet4f& v) {
    uint32_t values[4];
    _mm_store_ps(reinterpret_cast<float*>(values), v);
    
    std::cout << name << ": [";
    for (int i = 3; i >= 0; i--) {
        std::cout << "0x" << std::hex << std::setfill('0') << std::setw(8) << values[i];
        if (i > 0) std::cout << ", ";
    }
    std::cout << std::dec << "]\n";
}

// 演示基本功能
void demonstrate_basic_functionality() {
    std::cout << "=== psignbit 基本功能演示 ===\n\n";
    
    // 测试数据：包含正数、负数、零、无穷大
    Packet4f test_values = _mm_set_ps(-3.14f, 0.0f, 2.718f, -0.0f);
    
    std::cout << "输入数据:\n";
    print_float_vector("原始值", test_values);
    print_bit_pattern("位模式", test_values);
    
    // 提取符号位
    Packet4f sign_bits = psignbit(test_values);
    
    std::cout << "\n符号位提取结果:\n";
    print_float_vector("符号掩码", sign_bits);
    print_bit_pattern("掩码位模式", sign_bits);
    
    // 解释结果
    float values[4], signs[4];
    _mm_store_ps(values, test_values);
    _mm_store_ps(signs, sign_bits);
    
    std::cout << "\n详细解释:\n";
    for (int i = 3; i >= 0; i--) {
        std::cout << "值 " << std::setw(8) << std::fixed << std::setprecision(3) << values[i] 
                  << " -> 符号位: " << ((*reinterpret_cast<uint32_t*>(&signs[i]) == 0) ? "正" : "负")
                  << " (0x" << std::hex << *reinterpret_cast<uint32_t*>(&signs[i]) << std::dec << ")\n";
    }
}

// 演示逐步过程
void demonstrate_step_by_step() {
    std::cout << "\n=== 逐步过程演示 ===\n\n";
    
    Packet4f input = _mm_set_ps(-5.0f, 3.0f, -1.0f, 2.0f);  // [2.0, -1.0, 3.0, -5.0]
    
    std::cout << "输入: ";
    print_float_vector("", input);
    print_bit_pattern("位模式", input);
    
    // 步骤1: 转换为整数
    __m128i as_int = _mm_castps_si128(input);
    std::cout << "\n步骤1 - 转换为整数向量:\n";
    print_bit_pattern("整数位模式", _mm_castsi128_ps(as_int));
    
    // 步骤2: 算术右移31位
    __m128i shifted = _mm_srai_epi32(as_int, 31);
    std::cout << "\n步骤2 - 算术右移31位:\n";
    print_bit_pattern("右移结果", _mm_castsi128_ps(shifted));
    
    // 步骤3: 转换回浮点数
    Packet4f result = _mm_castsi128_ps(shifted);
    std::cout << "\n步骤3 - 最终结果:\n";
    print_float_vector("符号掩码", result);
    
    // 验证每个位置
    std::cout << "\n验证:\n";
    uint32_t input_bits[4], result_bits[4];
    _mm_store_ps(reinterpret_cast<float*>(input_bits), input);
    _mm_store_ps(reinterpret_cast<float*>(result_bits), result);
    
    for (int i = 3; i >= 0; i--) {
        bool is_negative = (input_bits[i] & 0x80000000) != 0;
        bool mask_set = result_bits[i] == 0xFFFFFFFF;
        
        std::cout << "位置" << (3-i) << ": "
                  << "原始=" << std::hex << input_bits[i] << " "
                  << "符号位=" << (is_negative ? "1" : "0") << " "
                  << "掩码=" << std::hex << result_bits[i] << " "
                  << "匹配=" << (is_negative == mask_set ? "✓" : "✗") << std::dec << "\n";
    }
}

// 实际应用示例
void demonstrate_applications() {
    std::cout << "\n=== 实际应用示例 ===\n\n";
    
    Packet4f values = _mm_set_ps(-2.5f, 1.8f, -0.7f, 3.2f);
    
    std::cout << "1. 绝对值计算:\n";
    print_float_vector("原始值", values);
    
    // 使用符号位来计算绝对值
    Packet4f sign_mask = psignbit(values);
    Packet4f abs_values = _mm_andnot_ps(sign_mask, values);  // 清除符号位
    
    print_float_vector("绝对值", abs_values);
    
    std::cout << "\n2. 符号复制:\n";
    Packet4f magnitudes = _mm_set_ps(4.0f, 3.0f, 2.0f, 1.0f);
    Packet4f signs = _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f);
    
    print_float_vector("幅度值", magnitudes);
    print_float_vector("符号源", signs);
    
    // 提取符号并应用到幅度
    Packet4f sign_bits = psignbit(signs);
    Packet4f abs_magnitudes = _mm_andnot_ps(_mm_set1_ps(-0.0f), magnitudes);  // 确保为正
    Packet4f signed_result = _mm_or_ps(abs_magnitudes, sign_bits);
    
    print_float_vector("应用符号后", signed_result);
    
    std::cout << "\n3. 条件选择（基于符号）:\n";
    Packet4f data = _mm_set_ps(-1.5f, 2.3f, -4.1f, 0.8f);
    Packet4f positive_replacement = _mm_set_ps(10.0f, 20.0f, 30.0f, 40.0f);
    
    print_float_vector("原始数据", data);
    print_float_vector("正数替换值", positive_replacement);
    
    // 负数保持不变，正数替换
    Packet4f is_negative = psignbit(data);
    Packet4f result = _mm_or_ps(
        _mm_and_ps(is_negative, data),                    // 负数：保持原值
        _mm_andnot_ps(is_negative, positive_replacement)  // 正数：使用替换值
    );
    
    print_float_vector("选择结果", result);
}

// 性能对比
void performance_comparison() {
    std::cout << "\n=== 性能对比 ===\n\n";
    
    const size_t num_operations = 10000000;
    std::vector<Packet4f> test_data;
    test_data.reserve(num_operations);
    
    // 生成测试数据
    for (size_t i = 0; i < num_operations; ++i) {
        float a = static_cast<float>(i) - num_operations/2;
        float b = static_cast<float>(i*2) - num_operations;
        float c = static_cast<float>(i*3) - num_operations*1.5f;
        float d = static_cast<float>(i*4) - num_operations*2;
        test_data.push_back(_mm_set_ps(a, b, c, d));
    }
    
    volatile Packet4f sink;
    
    // 方法1: psignbit (位操作)
    auto start = std::chrono::high_resolution_clock::now();
    for (const auto& data : test_data) {
        sink = psignbit(data);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto psignbit_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 方法2: 标量比较（逐个检查）
    start = std::chrono::high_resolution_clock::now();
    for (const auto& data : test_data) {
        float values[4];
        _mm_store_ps(values, data);
        
        uint32_t results[4];
        for (int i = 0; i < 4; ++i) {
            results[i] = (values[i] < 0.0f) ? 0xFFFFFFFF : 0x00000000;
        }
        
        sink = _mm_load_ps(reinterpret_cast<float*>(results));
    }
    end = std::chrono::high_resolution_clock::now();
    auto scalar_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 方法3: 使用比较指令
    start = std::chrono::high_resolution_clock::now();
    for (const auto& data : test_data) {
        sink = _mm_cmplt_ps(data, _mm_setzero_ps());  // data < 0
    }
    end = std::chrono::high_resolution_clock::now();
    auto compare_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "操作次数: " << num_operations << " × 4 = " << num_operations * 4 << "\n";
    std::cout << "psignbit (位操作):     " << psignbit_time.count() << " μs\n";
    std::cout << "标量比较:              " << scalar_time.count() << " μs (" 
              << std::fixed << std::setprecision(1) 
              << (double)scalar_time.count() / psignbit_time.count() << "x 慢)\n";
    std::cout << "SIMD比较指令:          " << compare_time.count() << " μs ("
              << (double)compare_time.count() / psignbit_time.count() << "x)\n";
    
    std::cout << "\n注意: psignbit提取实际符号位，比较指令检查是否<0 (结果略有不同)\n";
}

int main() {
    std::cout << "Eigen psignbit 函数分析\n";
    std::cout << "======================\n\n";
    
    demonstrate_basic_functionality();
    demonstrate_step_by_step();
    demonstrate_applications();
    performance_comparison();
    
    return 0;
}