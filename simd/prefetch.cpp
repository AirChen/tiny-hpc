#include <immintrin.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>

#pragma GCC target("avx")
#pragma GCC target("avx2")
#pragma GCC target("fma")
#pragma GCC optimize("O3")

// #define _MM_HINT_T0     3   // 预取到所有缓存级别 (最接近CPU)
// #define _MM_HINT_T1     2   // 预取到L2/L3缓存 (跳过L1)
// #define _MM_HINT_T2     1   // 预取到L3缓存 (跳过L1/L2)
// #define _MM_HINT_NTA    0   // 非时间局部性预取 (不污染缓存)

// ============================================================================
// 预取提示类型解释
// ============================================================================

void explain_prefetch_hints() {
    std::cout << "=== 内存预取提示类型 ===\n\n";
    
    std::cout << "_MM_HINT_T0  (3): 预取到所有缓存级别\n";
    std::cout << "  - 数据会被加载到L1, L2, L3缓存\n";
    std::cout << "  - 适合：即将频繁访问的热数据\n";
    std::cout << "  - 延迟：最低 (~1-4 cycles)\n\n";
    
    std::cout << "_MM_HINT_T1  (2): 预取到L2/L3缓存\n"; 
    std::cout << "  - 跳过L1缓存，直接加载到L2/L3\n";
    std::cout << "  - 适合：中等优先级数据\n";
    std::cout << "  - 延迟：中等 (~10-20 cycles)\n\n";
    
    std::cout << "_MM_HINT_T2  (1): 预取到L3缓存\n";
    std::cout << "  - 只加载到最后一级缓存(LLC)\n"; 
    std::cout << "  - 适合：低优先级或大数据块\n";
    std::cout << "  - 延迟：较高 (~30-50 cycles)\n\n";
    
    std::cout << "_MM_HINT_NTA (0): 非时间局部性预取\n";
    std::cout << "  - 不污染正常缓存层次结构\n";
    std::cout << "  - 适合：流式数据，只访问一次的数据\n";
    std::cout << "  - 延迟：变化很大\n\n";
}

// ============================================================================
// 基础预取示例
// ============================================================================

void demonstrate_basic_prefetch() {
    std::cout << "=== 基础预取使用示例 ===\n\n";
    
    const size_t array_size = 1024;
    std::vector<float> data(array_size);
    
    // 初始化数据
    std::iota(data.begin(), data.end(), 1.0f);
    
    std::cout << "示例1: 顺序访问 + 预取\n";
    std::cout << "----------------------\n";
    
    float sum = 0.0f;
    const int prefetch_distance = 64;  // 预取距离（元素个数）
    
    for (size_t i = 0; i < array_size; ++i) {
        // 预取未来的数据
        if (i + prefetch_distance < array_size) {
            _mm_prefetch(reinterpret_cast<const char*>(&data[i + prefetch_distance]), _MM_HINT_T0);
        }
        
        // 处理当前数据
        sum += data[i] * data[i];  // 一些计算工作
        
        if (i < 10) {  // 只打印前几个
            std::cout << "处理 data[" << i << "] = " << data[i];
            if (i + prefetch_distance < array_size) {
                std::cout << ", 预取 data[" << (i + prefetch_distance) << "]";
            }
            std::cout << "\n";
        }
    }
    
    std::cout << "...\n";
    std::cout << "计算结果: " << std::setprecision(2) << std::scientific << sum << "\n\n";
}

// ============================================================================
// 随机访问优化
// ============================================================================

void demonstrate_random_access_prefetch() {
    std::cout << "=== 随机访问模式的预取优化 ===\n\n";
    
    const size_t data_size = 10000;
    const size_t access_count = 1000;
    
    std::vector<float> data(data_size);
    std::vector<size_t> access_pattern(access_count);
    
    // 初始化数据
    std::iota(data.begin(), data.end(), 1.0f);
    
    // 生成随机访问模式
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dis(0, data_size - 1);
    
    for (size_t i = 0; i < access_count; ++i) {
        access_pattern[i] = dis(gen);
    }
    
    std::cout << "测试场景: 随机访问 " << access_count << " 次，数据大小 " << data_size << "\n\n";
    
    // 测试1: 无预取
    auto start = std::chrono::high_resolution_clock::now();
    volatile float sum1 = 0.0f;
    for (size_t i = 0; i < access_count; ++i) {
        sum1 += data[access_pattern[i]];
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto no_prefetch_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 测试2: 带预取（预测性预取）
    start = std::chrono::high_resolution_clock::now();
    volatile float sum2 = 0.0f;
    const int look_ahead = 8;  // 向前看8步
    
    for (size_t i = 0; i < access_count; ++i) {
        // 预取未来的访问
        if (i + look_ahead < access_count) {
            _mm_prefetch(reinterpret_cast<const char*>(&data[access_pattern[i + look_ahead]]), _MM_HINT_T0);
        }
        
        sum2 += data[access_pattern[i]];
    }
    end = std::chrono::high_resolution_clock::now();
    auto with_prefetch_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "结果对比:\n";
    std::cout << "无预取:   " << std::setw(8) << no_prefetch_time.count() << " μs\n";
    std::cout << "带预取:   " << std::setw(8) << with_prefetch_time.count() << " μs\n";
    std::cout << "性能提升: " << std::fixed << std::setprecision(1) 
              << (double)no_prefetch_time.count() / with_prefetch_time.count() << "x\n\n";
    
    std::cout << "注意: 随机访问的预取效果取决于访问模式的可预测性\n\n";
}

// ============================================================================
// 矩阵运算中的预取
// ============================================================================

void demonstrate_matrix_prefetch() {
    std::cout << "=== 矩阵运算中的预取优化 ===\n\n";
    
    const int N = 512;  // 矩阵大小
    std::vector<std::vector<float>> A(N, std::vector<float>(N));
    std::vector<std::vector<float>> B(N, std::vector<float>(N));
    std::vector<std::vector<float>> C1(N, std::vector<float>(N, 0.0f));  // 无预取结果
    std::vector<std::vector<float>> C2(N, std::vector<float>(N, 0.0f));  // 有预取结果
    
    // 初始化矩阵
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i][j] = dis(gen);
            B[i][j] = dis(gen);
        }
    }
    
    std::cout << "矩阵大小: " << N << "x" << N << "\n";
    std::cout << "计算 C = A * B\n\n";
    
    // 测试1: 标准矩阵乘法（无预取）
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                C1[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto no_prefetch_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // 测试2: 带预取的矩阵乘法
    start = std::chrono::high_resolution_clock::now();
    const int prefetch_rows_ahead = 2;  // 预取提前的行数
    
    for (int i = 0; i < N; ++i) {
        // 预取下一行的A数据
        if (i + prefetch_rows_ahead < N) {
            for (int k = 0; k < N; k += 8) {  // 每8个元素预取一次
                _mm_prefetch(reinterpret_cast<const char*>(&A[i + prefetch_rows_ahead][k]), _MM_HINT_T0);
            }
        }
        
        for (int j = 0; j < N; ++j) {
            // 预取B矩阵的下一列
            if (j + 1 < N) {
                for (int k = 0; k < N; k += 8) {
                    _mm_prefetch(reinterpret_cast<const char*>(&B[k][j + 1]), _MM_HINT_T1);
                }
            }
            
            for (int k = 0; k < N; ++k) {
                C2[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    end = std::chrono::high_resolution_clock::now();
    auto with_prefetch_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // 验证结果一致性
    bool results_match = true;
    for (int i = 0; i < N && results_match; ++i) {
        for (int j = 0; j < N && results_match; ++j) {
            if (std::abs(C1[i][j] - C2[i][j]) > 1e-5f) {
                results_match = false;
            }
        }
    }
    
    std::cout << "性能结果:\n";
    std::cout << "无预取:   " << std::setw(6) << no_prefetch_time.count() << " ms\n";
    std::cout << "带预取:   " << std::setw(6) << with_prefetch_time.count() << " ms\n";
    std::cout << "性能提升: " << std::fixed << std::setprecision(1)
              << (double)no_prefetch_time.count() / with_prefetch_time.count() << "x\n";
    std::cout << "结果正确性: " << (results_match ? "✓" : "✗") << "\n\n";
}

// ============================================================================
// 流式数据处理
// ============================================================================

void demonstrate_streaming_prefetch() {
    std::cout << "=== 流式数据处理中的预取 ===\n\n";
    
    const size_t stream_size = 1000000;  // 1M元素
    std::vector<float> input_stream(stream_size);
    std::vector<float> output_stream1(stream_size);
    std::vector<float> output_stream2(stream_size);
    
    // 初始化输入流
    for (size_t i = 0; i < stream_size; ++i) {
        input_stream[i] = std::sin(static_cast<float>(i) * 0.001f);
    }
    
    std::cout << "流式处理: 应用低通滤波器到 " << stream_size << " 个样本\n";
    std::cout << "滤波器: y[i] = 0.3*x[i] + 0.4*x[i-1] + 0.3*x[i-2]\n\n";
    
    // 测试1: 无预取版本
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 2; i < stream_size; ++i) {
        output_stream1[i] = 0.3f * input_stream[i] + 
                           0.4f * input_stream[i-1] + 
                           0.3f * input_stream[i-2];
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto no_prefetch_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 测试2: 使用NTA预取（流式访问）
    start = std::chrono::high_resolution_clock::now();
    const int prefetch_distance = 32;
    
    for (size_t i = 2; i < stream_size; ++i) {
        // 使用NTA预取未来的输入数据（一次性访问）
        if (i + prefetch_distance < stream_size) {
            _mm_prefetch(reinterpret_cast<const char*>(&input_stream[i + prefetch_distance]), _MM_HINT_NTA);
        }
        
        output_stream2[i] = 0.3f * input_stream[i] + 
                           0.4f * input_stream[i-1] + 
                           0.3f * input_stream[i-2];
    }
    end = std::chrono::high_resolution_clock::now();
    auto nta_prefetch_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 验证结果
    bool results_match = true;
    for (size_t i = 2; i < stream_size && results_match; ++i) {
        if (std::abs(output_stream1[i] - output_stream2[i]) > 1e-6f) {
            results_match = false;
        }
    }
    
    std::cout << "性能对比:\n";
    std::cout << "无预取:     " << std::setw(8) << no_prefetch_time.count() << " μs\n";
    std::cout << "NTA预取:    " << std::setw(8) << nta_prefetch_time.count() << " μs\n";
    std::cout << "性能变化:   " << std::fixed << std::setprecision(1)
              << (double)no_prefetch_time.count() / nta_prefetch_time.count() << "x\n";
    std::cout << "结果正确:   " << (results_match ? "✓" : "✗") << "\n\n";
    
    std::cout << "注意: NTA预取对流式数据特别有效，避免了缓存污染\n\n";
}

// ============================================================================
// 链表遍历中的预取
// ============================================================================

struct ListNode {
    float data;
    ListNode* next;
    char padding[64 - sizeof(float) - sizeof(ListNode*)];  // 缓存行对齐
};

void demonstrate_list_prefetch() {
    std::cout << "=== 链表遍历中的预取优化 ===\n\n";
    
    const size_t list_size = 10000;
    std::vector<ListNode> nodes(list_size);
    
    // 构建链表
    for (size_t i = 0; i < list_size - 1; ++i) {
        nodes[i].data = static_cast<float>(i);
        nodes[i].next = &nodes[i + 1];
    }
    nodes[list_size - 1].data = static_cast<float>(list_size - 1);
    nodes[list_size - 1].next = nullptr;
    
    std::cout << "链表大小: " << list_size << " 节点\n";
    std::cout << "节点大小: " << sizeof(ListNode) << " 字节 (缓存行对齐)\n\n";
    
    // 测试1: 标准链表遍历
    auto start = std::chrono::high_resolution_clock::now();
    volatile float sum1 = 0.0f;
    ListNode* current = &nodes[0];
    while (current) {
        sum1 += current->data;
        current = current->next;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto no_prefetch_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 测试2: 带预取的链表遍历
    start = std::chrono::high_resolution_clock::now();
    volatile float sum2 = 0.0f;
    current = &nodes[0];
    while (current) {
        // 预取下一个节点
        if (current->next) {
            _mm_prefetch(reinterpret_cast<const char*>(current->next), _MM_HINT_T0);
        }
        
        sum2 += current->data;
        current = current->next;
    }
    end = std::chrono::high_resolution_clock::now();
    auto with_prefetch_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "遍历结果对比:\n";
    std::cout << "无预取:   " << std::setw(8) << no_prefetch_time.count() << " μs\n";
    std::cout << "带预取:   " << std::setw(8) << with_prefetch_time.count() << " μs\n";
    std::cout << "性能提升: " << std::fixed << std::setprecision(1)
              << (double)no_prefetch_time.count() / with_prefetch_time.count() << "x\n";
    std::cout << "结果一致: " << (std::abs(sum1 - sum2) < 1e-6f ? "✓" : "✗") << "\n\n";
}

// ============================================================================
// 预取最佳实践指南
// ============================================================================

void show_best_practices() {
    std::cout << "=== 预取最佳实践指南 ===\n\n";
    
    std::cout << "1. 预取距离选择:\n";
    std::cout << "   - 太近: 数据可能已经在缓存中\n";
    std::cout << "   - 太远: 预取的数据可能被换出\n";
    std::cout << "   - 推荐: 64-256字节 (1-4个缓存行)\n\n";
    
    std::cout << "2. 预取提示选择:\n";
    std::cout << "   - 热数据、频繁访问 -> _MM_HINT_T0\n";
    std::cout << "   - 中等优先级数据   -> _MM_HINT_T1  \n";
    std::cout << "   - 大块、低优先级   -> _MM_HINT_T2\n";
    std::cout << "   - 流式、一次访问   -> _MM_HINT_NTA\n\n";
    
    std::cout << "3. 何时使用预取:\n";
    std::cout << "   ✓ 可预测的内存访问模式\n";
    std::cout << "   ✓ 内存带宽成为瓶颈\n"; 
    std::cout << "   ✓ 随机访问大数据集\n";
    std::cout << "   ✓ 链表、树等指针追踪\n";
    std::cout << "   ✗ 已经缓存友好的顺序访问\n";
    std::cout << "   ✗ 不可预测的随机模式\n";
    std::cout << "   ✗ 计算密集型（非内存绑定）\n\n";
    
    std::cout << "4. 代码示例模板:\n";
    std::cout << "```cpp\n";
    std::cout << "// 顺序访问预取\n";
    std::cout << "for (int i = 0; i < size; ++i) {\n";
    std::cout << "    if (i + PREFETCH_DISTANCE < size) {\n";
    std::cout << "        _mm_prefetch(&data[i + PREFETCH_DISTANCE], _MM_HINT_T0);\n";
    std::cout << "    }\n";
    std::cout << "    // 处理 data[i]\n";
    std::cout << "}\n\n";
    
    std::cout << "// 指针追踪预取\n";
    std::cout << "Node* current = head;\n";
    std::cout << "while (current) {\n";
    std::cout << "    if (current->next) {\n";
    std::cout << "        _mm_prefetch(current->next, _MM_HINT_T0);\n";
    std::cout << "    }\n";
    std::cout << "    // 处理 current\n";
    std::cout << "    current = current->next;\n";
    std::cout << "}\n";
    std::cout << "```\n";
}

int main() {
    std::cout << "内存预取 (_mm_prefetch) 详解与实践\n";
    std::cout << "==================================\n\n";
    
    explain_prefetch_hints();
    demonstrate_basic_prefetch();
    demonstrate_random_access_prefetch();
    demonstrate_matrix_prefetch();
    demonstrate_streaming_prefetch(); 
    demonstrate_list_prefetch();
    // show_best_practices();
    
    return 0;
}