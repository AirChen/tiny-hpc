#include <immintrin.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>

using Packet4f = __m128;
using Index = int;

// ============================================================================
// 基础工具函数
// ============================================================================

// 提取向量的第一个元素
inline float pfirst(const Packet4f& a) {
    return _mm_cvtss_f32(a);
}

// 打印向量
void print_vector(const char* name, const Packet4f& v) {
    float values[4];
    _mm_store_ps(values, v);
    std::cout << name << ": [";
    for (int i = 3; i >= 0; i--) {  // 按SSE存储顺序
        std::cout << std::setw(6) << std::fixed << std::setprecision(1) << values[i];
        if (i > 0) std::cout << ", ";
    }
    std::cout << "]\n";
}

// 打印数组的一部分
void print_array_section(const char* name, const float* arr, int start, int count, int stride = 1) {
    std::cout << name << ": ";
    for (int i = 0; i < count; ++i) {
        std::cout << std::setw(6) << std::fixed << std::setprecision(1) << arr[start + i * stride];
        if (i < count - 1) std::cout << ", ";
    }
    std::cout << "\n";
}

// ============================================================================
// Gather/Scatter 实现
// ============================================================================

// Gather操作：按步长收集元素
template<typename Scalar, typename Packet>
Packet4f pgather(const float* from, Index stride) {
    return _mm_set_ps(from[3 * stride], from[2 * stride], from[1 * stride], from[0 * stride]);
}

// Scatter操作：按步长分散元素
template<typename Scalar, typename Packet>
void pscatter(float* to, const Packet4f& from, Index stride) {
    to[stride * 0] = pfirst(from);
    to[stride * 1] = pfirst(_mm_shuffle_ps(from, from, 1));
    to[stride * 2] = pfirst(_mm_shuffle_ps(from, from, 2));
    to[stride * 3] = pfirst(_mm_shuffle_ps(from, from, 3));
}

// ============================================================================
// 基础演示
// ============================================================================

void demonstrate_basic_operations() {
    std::cout << "=== 基础 Gather/Scatter 操作演示 ===\n\n";
    
    // 创建测试数组
    std::vector<float> source = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 
        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f
    };
    
    std::cout << "源数组: ";
    for (size_t i = 0; i < source.size(); ++i) {
        std::cout << std::setw(4) << std::fixed << std::setprecision(0) << source[i];
    }
    std::cout << "\n";
    std::cout << "索引:   ";
    for (size_t i = 0; i < source.size(); ++i) {
        std::cout << std::setw(4) << i;
    }
    std::cout << "\n\n";
    
    // 测试不同步长的gather操作
    std::cout << "Gather操作测试:\n";
    std::cout << "---------------\n";
    
    for (int stride = 1; stride <= 4; ++stride) {
        std::cout << "步长 " << stride << ": 从索引 [0, " << stride << ", " << 2*stride 
                  << ", " << 3*stride << "] 收集\n";
        
        if (3 * stride < static_cast<int>(source.size())) {
            Packet4f gathered = pgather<float, Packet4f>(source.data(), stride);
            print_vector("  结果", gathered);
            
            // 显示收集的具体值
            std::cout << "  元素: " << source[0] << ", " << source[stride] 
                      << ", " << source[2*stride] << ", " << source[3*stride] << "\n";
        } else {
            std::cout << "  (超出数组边界)\n";
        }
        std::cout << "\n";
    }
}

void demonstrate_scatter_operations() {
    std::cout << "=== Scatter操作演示 ===\n\n";
    
    // 创建向量数据
    Packet4f data = _mm_set_ps(40.0f, 30.0f, 20.0f, 10.0f);  // [10, 20, 30, 40]
    print_vector("输入向量", data);
    
    // 测试不同步长的scatter操作
    for (int stride = 1; stride <= 3; ++stride) {
        std::cout << "\n步长 " << stride << " 的scatter操作:\n";
        
        std::vector<float> target(16, 0.0f);  // 用0填充
        pscatter<float, Packet4f>(target.data(), data, stride);
        
        std::cout << "目标数组: ";
        for (size_t i = 0; i < std::min(target.size(), size_t(12)); ++i) {
            if (target[i] != 0.0f) {
                std::cout << std::setw(4) << std::fixed << std::setprecision(0) << target[i];
            } else {
                std::cout << "   0";
            }
        }
        std::cout << "\n";
        
        std::cout << "写入位置: [0, " << stride << ", " << 2*stride << ", " << 3*stride << "]\n";
    }
}

// ============================================================================
// 实际应用场景
// ============================================================================

void demonstrate_matrix_operations() {
    std::cout << "\n=== 矩阵操作应用 ===\n\n";
    
    // 4x4矩阵，按行存储
    std::vector<float> matrix = {
        1.0f,  2.0f,  3.0f,  4.0f,    // 第0行
        5.0f,  6.0f,  7.0f,  8.0f,    // 第1行
        9.0f,  10.0f, 11.0f, 12.0f,   // 第2行
        13.0f, 14.0f, 15.0f, 16.0f    // 第3行
    };
    
    std::cout << "原始4x4矩阵 (按行存储):\n";
    for (int i = 0; i < 4; ++i) {
        std::cout << "  ";
        for (int j = 0; j < 4; ++j) {
            std::cout << std::setw(6) << std::fixed << std::setprecision(0) << matrix[i * 4 + j];
        }
        std::cout << "\n";
    }
    
    std::cout << "\n1. 提取列 (gather):\n";
    for (int col = 0; col < 4; ++col) {
        // 从每行的第col列收集元素，步长为4（跳过整行）
        Packet4f column = pgather<float, Packet4f>(matrix.data() + col, 4);
        std::cout << "  第" << col << "列: ";
        print_vector("", column);
    }
    
    std::cout << "\n2. 矩阵转置 (gather + scatter):\n";
    std::vector<float> transposed(16, 0.0f);
    
    for (int col = 0; col < 4; ++col) {
        // 收集第col列
        Packet4f column = pgather<float, Packet4f>(matrix.data() + col, 4);
        // 散布到转置矩阵的第col行
        pscatter<float, Packet4f>(transposed.data() + col * 4, column, 1);
    }
    
    std::cout << "  转置后的矩阵:\n";
    for (int i = 0; i < 4; ++i) {
        std::cout << "    ";
        for (int j = 0; j < 4; ++j) {
            std::cout << std::setw(6) << std::fixed << std::setprecision(0) << transposed[i * 4 + j];
        }
        std::cout << "\n";
    }
}

void demonstrate_interleaved_data() {
    std::cout << "\n=== 交错数据处理 ===\n\n";
    
    // RGB像素数据 (交错存储)
    std::vector<float> rgb_data = {
        1.0f, 2.0f, 3.0f,    // 像素0: R, G, B
        4.0f, 5.0f, 6.0f,    // 像素1: R, G, B  
        7.0f, 8.0f, 9.0f,    // 像素2: R, G, B
        10.0f, 11.0f, 12.0f, // 像素3: R, G, B
        13.0f, 14.0f, 15.0f, // 像素4: R, G, B
        16.0f, 17.0f, 18.0f  // 像素5: R, G, B
    };
    
    std::cout << "交错的RGB数据:\n";
    for (size_t i = 0; i < rgb_data.size(); i += 3) {
        std::cout << "  像素" << i/3 << ": R=" << rgb_data[i] 
                  << " G=" << rgb_data[i+1] << " B=" << rgb_data[i+2] << "\n";
    }
    
    std::cout << "\n提取各颜色通道 (使用gather):\n";
    
    // 提取R通道：从索引0开始，步长3
    Packet4f red_channel = pgather<float, Packet4f>(rgb_data.data() + 0, 3);
    print_vector("R通道", red_channel);
    
    // 提取G通道：从索引1开始，步长3  
    Packet4f green_channel = pgather<float, Packet4f>(rgb_data.data() + 1, 3);
    print_vector("G通道", green_channel);
    
    // 提取B通道：从索引2开始，步长3
    Packet4f blue_channel = pgather<float, Packet4f>(rgb_data.data() + 2, 3);
    print_vector("B通道", blue_channel);
    
    // 处理后重新写回 (例如：增强亮度)
    std::cout << "\n亮度增强后写回 (使用scatter):\n";
    Packet4f enhanced_red = _mm_mul_ps(red_channel, _mm_set1_ps(1.2f));   // 增强20%
    Packet4f enhanced_green = _mm_mul_ps(green_channel, _mm_set1_ps(1.1f)); // 增强10%
    Packet4f enhanced_blue = _mm_mul_ps(blue_channel, _mm_set1_ps(1.3f));  // 增强30%
    
    std::vector<float> enhanced_rgb = rgb_data; // 复制原数据
    
    pscatter<float, Packet4f>(enhanced_rgb.data() + 0, enhanced_red, 3);   // 写回R通道
    pscatter<float, Packet4f>(enhanced_rgb.data() + 1, enhanced_green, 3); // 写回G通道
    pscatter<float, Packet4f>(enhanced_rgb.data() + 2, enhanced_blue, 3);  // 写回B通道
    
    std::cout << "增强后的RGB数据:\n";
    for (size_t i = 0; i < enhanced_rgb.size(); i += 3) {
        std::cout << "  像素" << i/3 << ": R=" << std::setprecision(1) << enhanced_rgb[i]
                  << " G=" << enhanced_rgb[i+1] << " B=" << enhanced_rgb[i+2] << "\n";
    }
}

// ============================================================================
// 性能对比
// ============================================================================

void performance_comparison() {
    std::cout << "\n=== 性能对比 ===\n\n";
    
    const size_t data_size = 1000000;
    const int stride = 4;
    std::vector<float> source(data_size);
    std::vector<float> target_simd(data_size, 0.0f);
    std::vector<float> target_scalar(data_size, 0.0f);
    
    // 初始化测试数据
    for (size_t i = 0; i < data_size; ++i) {
        source[i] = static_cast<float>(i);
    }
    
    const size_t num_operations = data_size / (stride * 4);  // 每次处理4个元素
    
    volatile float sink = 0.0f;  // 防止编译器优化
    
    std::cout << "测试配置:\n";
    std::cout << "  数据大小: " << data_size << " 个float\n";
    std::cout << "  步长: " << stride << "\n";
    std::cout << "  操作次数: " << num_operations << "\n\n";
    
    // 测试1: SIMD gather
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_operations; ++i) {
        Packet4f gathered = pgather<float, Packet4f>(source.data() + i * stride * 4, stride);
        sink += pfirst(gathered);  // 防止优化
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto simd_gather_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 测试2: 标量gather  
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_operations; ++i) {
        size_t base = i * stride * 4;
        float values[4];
        values[0] = source[base + 0 * stride];
        values[1] = source[base + 1 * stride];
        values[2] = source[base + 2 * stride]; 
        values[3] = source[base + 3 * stride];
        sink += values[0];  // 防止优化
    }
    end = std::chrono::high_resolution_clock::now();
    auto scalar_gather_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 测试3: SIMD scatter
    Packet4f test_data = _mm_set_ps(4.0f, 3.0f, 2.0f, 1.0f);
    
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_operations; ++i) {
        pscatter<float, Packet4f>(target_simd.data() + i * stride * 4, test_data, stride);
    }
    end = std::chrono::high_resolution_clock::now();
    auto simd_scatter_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 测试4: 标量scatter
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_operations; ++i) {
        size_t base = i * stride * 4;
        target_scalar[base + 0 * stride] = 1.0f;
        target_scalar[base + 1 * stride] = 2.0f;
        target_scalar[base + 2 * stride] = 3.0f;
        target_scalar[base + 3 * stride] = 4.0f;
    }
    end = std::chrono::high_resolution_clock::now();
    auto scalar_scatter_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 输出结果
    std::cout << "性能结果:\n";
    std::cout << "--------\n";
    std::cout << "SIMD Gather:    " << std::setw(8) << simd_gather_time.count() << " μs\n";
    std::cout << "标量 Gather:    " << std::setw(8) << scalar_gather_time.count() << " μs  ";
    std::cout << "(" << std::fixed << std::setprecision(1) 
              << (double)scalar_gather_time.count() / simd_gather_time.count() << "x)\n";
    
    std::cout << "SIMD Scatter:   " << std::setw(8) << simd_scatter_time.count() << " μs\n";
    std::cout << "标量 Scatter:   " << std::setw(8) << scalar_scatter_time.count() << " μs  ";
    std::cout << "(" << (double)scalar_scatter_time.count() / simd_scatter_time.count() << "x)\n";
    
    std::cout << "\n注意: gather/scatter操作的性能很大程度上取决于内存访问模式和缓存性能\n";
    std::cout << "      在某些情况下，标量版本可能由于更好的缓存局部性而表现更好\n";
}

int main() {
    std::cout << "Eigen Gather/Scatter 操作详解\n";
    std::cout << "=============================\n\n";
    
    demonstrate_basic_operations();
    demonstrate_scatter_operations();
    demonstrate_matrix_operations();
    demonstrate_interleaved_data();
    performance_comparison();
    
    return 0;
}