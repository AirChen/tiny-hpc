#include <cstdlib>
#include <stdalign.h>
#include <immintrin.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <cmath>
#include <random>

#pragma GCC target("avx")
#pragma GCC target("avx2")
#pragma GCC target("fma")
#pragma GCC optimize("O3")

template<typename T>
class aligned_allocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    
    static constexpr size_t alignment = 32; // AVX需要32字节对齐
    
    aligned_allocator() = default;
    
    template<typename U>
    aligned_allocator(const aligned_allocator<U>&) {}
    
    pointer allocate(size_type n) {
        if (n == 0) return nullptr;
        
        void* ptr = nullptr;
        size_t size = n * sizeof(T);
        
#ifdef _WIN32
        ptr = _aligned_malloc(size, alignment);
#else
        if (posix_memalign(&ptr, alignment, size) != 0) {
            ptr = nullptr;
        }
#endif
        
        if (!ptr) {
            throw std::bad_alloc();
        }
        
        return static_cast<pointer>(ptr);
    }
    
    void deallocate(pointer p, size_type) {
        if (p) {
#ifdef _WIN32
            _aligned_free(p);
#else
            free(p);
#endif
        }
    }
    
    template<typename U>
    bool operator==(const aligned_allocator<U>&) const { return true; }
    
    template<typename U>
    bool operator!=(const aligned_allocator<U>&) const { return false; }
};
using aligned_vector_float = std::vector<float, aligned_allocator<float>>;

// 点结构定义
struct Point2D {
    float x, y;
    Point2D(float x = 0, float y = 0) : x(x), y(y) {}
};

struct Point3D {
    float x, y, z;
    Point3D(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
};

// ============================================================================
// 2D点距离计算
// ============================================================================

// 标量版本 - 计算连续点之间的距离
void compute_distances_2d_scalar(const std::vector<Point2D>& points, 
                                 std::vector<float>& distances) {
    size_t n = points.size();
    distances.resize(n - 1);
    
    for (size_t i = 0; i < n - 1; ++i) {
        float dx = points[i+1].x - points[i].x;
        float dy = points[i+1].y - points[i].y;
        distances[i] = std::sqrt(dx * dx + dy * dy);
    }
}

// SIMD版本 - 一次处理4对点
void compute_distances_2d_simd(const std::vector<Point2D>& points, 
                               std::vector<float>& distances) {
    size_t n = points.size();
    distances.resize(n - 1);
    
    size_t i = 0;
    // 处理4个点对为一组
    for (; i + 4 < n; i += 4) {
        // 加载8个点的坐标
        // points[i], points[i+1], points[i+2], points[i+3]
        // points[i+1], points[i+2], points[i+3], points[i+4]
        
        // 方法1：使用两个__m128加载x坐标
        __m128 x1 = _mm_set_ps(points[i+3].x, points[i+2].x, 
                              points[i+1].x, points[i].x);
        __m128 x2 = _mm_set_ps(points[i+4].x, points[i+3].x, 
                              points[i+2].x, points[i+1].x);
        
        __m128 y1 = _mm_set_ps(points[i+3].y, points[i+2].y, 
                              points[i+1].y, points[i].y);
        __m128 y2 = _mm_set_ps(points[i+4].y, points[i+3].y, 
                              points[i+2].y, points[i+1].y);
        
        // 计算差值
        __m128 dx = _mm_sub_ps(x2, x1);
        __m128 dy = _mm_sub_ps(y2, y1);
        
        // 计算平方
        __m128 dx2 = _mm_mul_ps(dx, dx);
        __m128 dy2 = _mm_mul_ps(dy, dy);
        
        // 计算距离平方
        __m128 dist2 = _mm_add_ps(dx2, dy2);
        
        // 计算平方根
        __m128 dist = _mm_sqrt_ps(dist2);
        
        // 存储结果
        _mm_store_ps(&distances[i], dist);
    }
    
    // 处理剩余的点
    for (; i < n - 1; ++i) {
        float dx = points[i+1].x - points[i].x;
        float dy = points[i+1].y - points[i].y;
        distances[i] = std::sqrt(dx * dx + dy * dy);
    }
}

// SIMD版本 - 一次处理8对点
void compute_distances_2d_simd_8f(const std::vector<Point2D>& points, 
                               std::vector<float>& distances) {
    size_t n = points.size();
    distances.resize(n - 1);
    
    size_t i = 0;
    // 处理8个点对为一组
    for (; i + 8 < n; i += 8) {
        // 方法1：使用两个__m128加载x坐标
        __m256 x1 = _mm256_set_ps(points[i+7].x, points[i+6].x, points[i+5].x, points[i+4].x, points[i+3].x, points[i+2].x, 
                              points[i+1].x, points[i].x);
        __m256 x2 = _mm256_set_ps(points[i+8].x, points[i+7].x, points[i+6].x, points[i+5].x, points[i+4].x, points[i+3].x, 
                              points[i+2].x, points[i+1].x);
        
        __m256 y1 = _mm256_set_ps(points[i+7].y, points[i+6].y, points[i+5].y, points[i+4].y, points[i+3].y, points[i+2].y, 
                              points[i+1].y, points[i].y);
        __m256 y2 = _mm256_set_ps(points[i+8].y, points[i+7].y, points[i+6].y, points[i+5].y, points[i+4].y, points[i+3].y, 
                              points[i+2].y, points[i+1].y);
        
        // 计算差值
        __m256 dx = _mm256_sub_ps(x2, x1);
        __m256 dy = _mm256_sub_ps(y2, y1);
        
        // 计算平方
        __m256 dx2 = _mm256_mul_ps(dx, dx);
        __m256 dy2 = _mm256_mul_ps(dy, dy);
        
        // 计算距离平方
        __m256 dist2 = _mm256_add_ps(dx2, dy2);
        
        // 计算平方根
        __m256 dist = _mm256_sqrt_ps(dist2);
        
        // 存储结果
        _mm256_storeu_ps(&distances[i], dist); // 不对齐版本 __256_soreu_ps 
    }
    
    // 处理剩余的点
    for (; i < n - 1; ++i) {
        float dx = points[i+1].x - points[i].x;
        float dy = points[i+1].y - points[i].y;
        distances[i] = std::sqrt(dx * dx + dy * dy);
    }
}

void compute_distances_2d_simd_8f(const std::vector<Point2D>& points, 
                               float distances[]) {
    size_t n = points.size();
    size_t i = 0;
    // 处理8个点对为一组
    for (; i + 8 < n; i += 8) {
        // 方法1：使用两个__m128加载x坐标
        __m256 x1 = _mm256_set_ps(points[i+7].x, points[i+6].x, points[i+5].x, points[i+4].x, points[i+3].x, points[i+2].x, 
                              points[i+1].x, points[i].x);
        __m256 x2 = _mm256_set_ps(points[i+8].x, points[i+7].x, points[i+6].x, points[i+5].x, points[i+4].x, points[i+3].x, 
                              points[i+2].x, points[i+1].x);
        
        __m256 y1 = _mm256_set_ps(points[i+7].y, points[i+6].y, points[i+5].y, points[i+4].y, points[i+3].y, points[i+2].y, 
                              points[i+1].y, points[i].y);
        __m256 y2 = _mm256_set_ps(points[i+8].y, points[i+7].y, points[i+6].y, points[i+5].y, points[i+4].y, points[i+3].y, 
                              points[i+2].y, points[i+1].y);
        
        // 计算差值
        __m256 dx = _mm256_sub_ps(x2, x1);
        __m256 dy = _mm256_sub_ps(y2, y1);
        
        // 计算平方
        __m256 dx2 = _mm256_mul_ps(dx, dx);
        __m256 dy2 = _mm256_mul_ps(dy, dy);
        
        // 计算距离平方
        __m256 dist2 = _mm256_add_ps(dx2, dy2);
        
        // 计算平方根
        __m256 dist = _mm256_sqrt_ps(dist2);
        
        // 存储结果
        _mm256_store_ps(&distances[i], dist); // 不对齐版本 __256_soreu_ps 
    }
    
    // 处理剩余的点
    for (; i < n - 1; ++i) {
        float dx = points[i+1].x - points[i].x;
        float dy = points[i+1].y - points[i].y;
        distances[i] = std::sqrt(dx * dx + dy * dy);
    }
}

// ============================================================================
// 3D点距离计算
// ============================================================================

// 标量版本
void compute_distances_3d_scalar(const std::vector<Point3D>& points, 
                                 std::vector<float>& distances) {
    size_t n = points.size();
    distances.resize(n - 1);
    
    for (size_t i = 0; i < n - 1; ++i) {
        float dx = points[i+1].x - points[i].x;
        float dy = points[i+1].y - points[i].y;
        float dz = points[i+1].z - points[i].z;
        distances[i] = std::sqrt(dx * dx + dy * dy + dz * dz);
    }
}

// SIMD版本 - 3D点距离
void compute_distances_3d_simd(const std::vector<Point3D>& points, 
                               std::vector<float>& distances) {
    size_t n = points.size();
    distances.resize(n - 1);
    
    size_t i = 0;
    // 一次处理4个点对
    for (; i + 4 < n; i += 4) {
        // 加载坐标
        __m128 x1 = _mm_set_ps(points[i+3].x, points[i+2].x, 
                              points[i+1].x, points[i].x);
        __m128 x2 = _mm_set_ps(points[i+4].x, points[i+3].x, 
                              points[i+2].x, points[i+1].x);
        
        __m128 y1 = _mm_set_ps(points[i+3].y, points[i+2].y, 
                              points[i+1].y, points[i].y);
        __m128 y2 = _mm_set_ps(points[i+4].y, points[i+3].y, 
                              points[i+2].y, points[i+1].y);
        
        __m128 z1 = _mm_set_ps(points[i+3].z, points[i+2].z, 
                              points[i+1].z, points[i].z);
        __m128 z2 = _mm_set_ps(points[i+4].z, points[i+3].z, 
                              points[i+2].z, points[i+1].z);
        
        // 计算差值和平方
        __m128 dx = _mm_sub_ps(x2, x1);
        __m128 dy = _mm_sub_ps(y2, y1);
        __m128 dz = _mm_sub_ps(z2, z1);
        
        __m128 dx2 = _mm_mul_ps(dx, dx);
        __m128 dy2 = _mm_mul_ps(dy, dy);
        __m128 dz2 = _mm_mul_ps(dz, dz);
        
        // 计算距离平方和距离
        __m128 dist2 = _mm_add_ps(_mm_add_ps(dx2, dy2), dz2);
        __m128 dist = _mm_sqrt_ps(dist2);
        
        // 存储结果
        _mm_store_ps(&distances[i], dist);
    }
    
    // 处理剩余的点
    for (; i < n - 1; ++i) {
        float dx = points[i+1].x - points[i].x;
        float dy = points[i+1].y - points[i].y;
        float dz = points[i+1].z - points[i].z;
        distances[i] = std::sqrt(dx * dx + dy * dy + dz * dz);
    }
}

// ============================================================================
// 优化版本 - SOA (Structure of Arrays) 数据布局
// ============================================================================

// SOA版本的点数据存储
struct Points2D_SOA {
    float* x;
    float* y;
    size_t capcity{0};
    
    void resize(size_t n) {
        x = (float*)aligned_alloc(64, n * sizeof(float));
        y = (float*)aligned_alloc(64, n * sizeof(float));
    }
    
    void push_back(float px, float py) {
        x[capcity] = px;
        y[capcity] = py;
        capcity++;
    }

    ~Points2D_SOA() {
        free(x);
        free(y);
    }
};

// SOA版本 - 更高效的SIMD计算
void compute_distances_2d_soa_simd(const Points2D_SOA& points, 
                                   std::vector<float>& distances) {
    size_t n = points.capcity;
    distances.resize(n - 1);
    
    size_t i = 0;
    // 一次处理4个点对
    for (; i + 4 < n; i += 4) {
        // 直接加载连续的坐标数据
        __m128 x1 = _mm_loadu_ps(&points.x[i]);      // x[i], x[i+1], x[i+2], x[i+3]
        __m128 x2 = _mm_loadu_ps(&points.x[i + 1]);  // x[i+1], x[i+2], x[i+3], x[i+4]
        
        __m128 y1 = _mm_loadu_ps(&points.y[i]);      // y[i], y[i+1], y[i+2], y[i+3]
        __m128 y2 = _mm_loadu_ps(&points.y[i + 1]);  // y[i+1], y[i+2], y[i+3], y[i+4]
        
        // 计算距离
        __m128 dx = _mm_sub_ps(x2, x1);
        __m128 dy = _mm_sub_ps(y2, y1);
        
        __m128 dx2 = _mm_mul_ps(dx, dx);
        __m128 dy2 = _mm_mul_ps(dy, dy);
        
        __m128 dist2 = _mm_add_ps(dx2, dy2);
        __m128 dist = _mm_sqrt_ps(dist2);
        
        // 存储结果
        _mm_storeu_ps(&distances[i], dist);
    }
    
    // 处理剩余的点
    for (; i < n - 1; ++i) {
        float dx = points.x[i+1] - points.x[i];
        float dy = points.y[i+1] - points.y[i];
        distances[i] = std::sqrt(dx * dx + dy * dy);
    }
}

// SOA版本 - 更高效的SIMD计算 8 float
void compute_distances_2d_soa_simd_8f(const Points2D_SOA& points, 
                                   std::vector<float>& distances) {
    size_t n = points.capcity;
    distances.resize(n - 1);
    
    size_t i = 0;
    // 一次处理8个点对
    for (; i + 8 < n; i += 8) {
        // 直接加载连续的坐标数据
        __m256 x1 = _mm256_loadu_ps(&points.x[i]);      // x[i], x[i+1], x[i+2], x[i+3]
        __m256 x2 = _mm256_loadu_ps(&points.x[i + 1]);  // x[i+1], x[i+2], x[i+3], x[i+4]
        
        __m256 y1 = _mm256_loadu_ps(&points.y[i]);      // y[i], y[i+1], y[i+2], y[i+3]
        __m256 y2 = _mm256_loadu_ps(&points.y[i + 1]);  // y[i+1], y[i+2], y[i+3], y[i+4]
        
        // 计算距离
        __m256 dx = _mm256_sub_ps(x2, x1);
        __m256 dy = _mm256_sub_ps(y2, y1);
        
        __m256 dx2 = _mm256_mul_ps(dx, dx);
        __m256 dy2 = _mm256_mul_ps(dy, dy);
        
        __m256 dist2 = _mm256_add_ps(dx2, dy2);
        __m256 dist = _mm256_sqrt_ps(dist2);
        
        // 存储结果
        _mm256_storeu_ps(&distances[i], dist);
    }
    
    // 处理剩余的点
    for (; i < n - 1; ++i) {
        float dx = points.x[i+1] - points.x[i];
        float dy = points.y[i+1] - points.y[i];
        distances[i] = std::sqrt(dx * dx + dy * dy);
    }
}

void compute_distances_2d_soa_simd_8f(const Points2D_SOA& points, 
                                   float* distances) {
    size_t n = points.capcity;
    size_t i = 0;
    // 一次处理8个点对
    for (; i + 8 < n; i += 8) {
        // 直接加载连续的坐标数据
        __m256 x1 = _mm256_loadu_ps(&points.x[i]);      // x[i], x[i+1], x[i+2], x[i+3]
        __m256 x2 = _mm256_loadu_ps(&points.x[i + 1]);  // x[i+1], x[i+2], x[i+3], x[i+4]
        
        __m256 y1 = _mm256_loadu_ps(&points.y[i]);      // y[i], y[i+1], y[i+2], y[i+3]
        __m256 y2 = _mm256_loadu_ps(&points.y[i + 1]);  // y[i+1], y[i+2], y[i+3], y[i+4]
        
        // 计算距离
        __m256 dx = _mm256_sub_ps(x2, x1);
        __m256 dy = _mm256_sub_ps(y2, y1);
        
        __m256 dx2 = _mm256_mul_ps(dx, dx);
        __m256 dy2 = _mm256_mul_ps(dy, dy);
        
        __m256 dist2 = _mm256_add_ps(dx2, dy2);
        __m256 dist = _mm256_sqrt_ps(dist2);
        
        // 存储结果
        _mm256_store_ps(&distances[i], dist);
    }
    
    // 处理剩余的点
    for (; i < n - 1; ++i) {
        float dx = points.x[i+1] - points.x[i];
        float dy = points.y[i+1] - points.y[i];
        distances[i] = std::sqrt(dx * dx + dy * dy);
    }
}
void compute_distances_2d_soa_simd_8f(const float* input_x, const float* input_y, size_t size_n, 
                                   float* distances) {
    size_t n = size_n;
    size_t i = 0;
    // 一次处理8个点对
    for (; i + 8 < n; i += 8) {
        // 直接加载连续的坐标数据
        __m256 x1 = _mm256_load_ps(&input_x[i]);      // x[i], x[i+1], x[i+2], x[i+3]
        __m256 x2 = _mm256_load_ps(&input_x[i]);  // x[i+1], x[i+2], x[i+3], x[i+4]
        
        __m256 y1 = _mm256_load_ps(&input_y[i]);      // y[i], y[i+1], y[i+2], y[i+3]
        __m256 y2 = _mm256_load_ps(&input_y[i]);  // y[i+1], y[i+2], y[i+3], y[i+4]

        // 这个地方使用对齐 不能 + 1 了
        
        // 计算距离
        __m256 dx = _mm256_sub_ps(x2, x1);
        __m256 dy = _mm256_sub_ps(y2, y1);
        
        __m256 dx2 = _mm256_mul_ps(dx, dx);
        __m256 dy2 = _mm256_mul_ps(dy, dy);
        
        __m256 dist2 = _mm256_add_ps(dx2, dy2);
        __m256 dist = _mm256_sqrt_ps(dist2);
        
        // 存储结果
        _mm256_store_ps(&distances[i], dist);
    }
    
    // 处理剩余的点
    for (; i < n - 1; ++i) {
        float dx = input_x[i+1] - input_x[i];
        float dy = input_y[i+1] - input_y[i];
        distances[i] = std::sqrt(dx * dx + dy * dy);
    }
}

// ============================================================================
// 高级优化：批量距离计算（一次计算更多距离）
// ============================================================================

// 计算点到多个其他点的距离
void compute_point_to_points_distance_simd(const Point2D& origin,
                                           const std::vector<Point2D>& targets,
                                           std::vector<float>& distances) {
    size_t n = targets.size();
    distances.resize(n);
    
    // 将原点坐标广播到SIMD寄存器
    __m128 origin_x = _mm_set1_ps(origin.x);
    __m128 origin_y = _mm_set1_ps(origin.y);
    
    size_t i = 0;
    // 一次处理4个目标点
    for (; i + 4 <= n; i += 4) {
        __m128 target_x = _mm_set_ps(targets[i+3].x, targets[i+2].x,
                                    targets[i+1].x, targets[i].x);
        __m128 target_y = _mm_set_ps(targets[i+3].y, targets[i+2].y,
                                    targets[i+1].y, targets[i].y);
        
        __m128 dx = _mm_sub_ps(target_x, origin_x);
        __m128 dy = _mm_sub_ps(target_y, origin_y);
        
        __m128 dx2 = _mm_mul_ps(dx, dx);
        __m128 dy2 = _mm_mul_ps(dy, dy);
        
        __m128 dist2 = _mm_add_ps(dx2, dy2);
        __m128 dist = _mm_sqrt_ps(dist2);
        
        _mm_store_ps(&distances[i], dist);
    }
    
    // 处理剩余的点
    for (; i < n; ++i) {
        float dx = targets[i].x - origin.x;
        float dy = targets[i].y - origin.y;
        distances[i] = std::sqrt(dx * dx + dy * dy);
    }
}

// ============================================================================
// 快速距离计算（省略平方根）
// ============================================================================

// 有时只需要比较距离大小，可以省略平方根计算
void compute_squared_distances_2d_simd(const std::vector<Point2D>& points, 
                                       std::vector<float>& squared_distances) {
    size_t n = points.size();
    squared_distances.resize(n - 1);
    
    size_t i = 0;
    for (; i + 4 < n; i += 4) {
        __m128 x1 = _mm_set_ps(points[i+3].x, points[i+2].x, 
                              points[i+1].x, points[i].x);
        __m128 x2 = _mm_set_ps(points[i+4].x, points[i+3].x, 
                              points[i+2].x, points[i+1].x);
        
        __m128 y1 = _mm_set_ps(points[i+3].y, points[i+2].y, 
                              points[i+1].y, points[i].y);
        __m128 y2 = _mm_set_ps(points[i+4].y, points[i+3].y, 
                              points[i+2].y, points[i+1].y);
        
        __m128 dx = _mm_sub_ps(x2, x1);
        __m128 dy = _mm_sub_ps(y2, y1);
        
        __m128 dx2 = _mm_mul_ps(dx, dx);
        __m128 dy2 = _mm_mul_ps(dy, dy);
        
        __m128 dist2 = _mm_add_ps(dx2, dy2);
        // 省略sqrt步骤
        
        _mm_store_ps(&squared_distances[i], dist2);
    }
    
    for (; i < n - 1; ++i) {
        float dx = points[i+1].x - points[i].x;
        float dy = points[i+1].y - points[i].y;
        squared_distances[i] = dx * dx + dy * dy;
    }
}

// ============================================================================
// 测试和性能对比
// ============================================================================

// 生成随机点
std::vector<Point2D> generate_random_points_2d(size_t count) {
    std::vector<Point2D> points;
    points.reserve(count);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1000.0f, 1000.0f);
    
    for (size_t i = 0; i < count; ++i) {
        points.emplace_back(dis(gen), dis(gen));
    }
    
    return points;
}

// 性能测试
void benchmark_distance_calculation() {
    const size_t num_points = 10000000;
    auto points = generate_random_points_2d(num_points);
    
    // 转换为SOA格式
    Points2D_SOA points_soa;
    points_soa.resize(num_points);
    float* input_x = (float*)aligned_alloc(64, num_points * sizeof(float));
    float* input_y = (float*)aligned_alloc(64, num_points * sizeof(float));
    for (size_t i = 0; i < num_points; ++i) {
        points_soa.push_back(points[i].x, points[i].y);
        
        input_x[i] = points[i].x;
        input_y[i] = points[i].y;
    }
    
    std::vector<float> distances_scalar, distances_simd, distances_soa;
    std::vector<float> distances_simd_8f, distances_soa_8f;
    // float* distances_simd_8f_ori = (float*)calloc(num_points, sizeof(float));
    // float* distances_soa_8f_ori = (float*)calloc(num_points, sizeof(float));
    float* distances_simd_8f_ori = (float*)aligned_alloc(64, num_points * sizeof(float));
    float* distances_soa_8f_ori = (float*)aligned_alloc(64, num_points * sizeof(float));
    float* distances_soa_8f_ori_opt = (float*)aligned_alloc(64, num_points * sizeof(float));

    // 测试标量版本
    auto start = std::chrono::high_resolution_clock::now();
    compute_distances_2d_scalar(points, distances_scalar);
    auto end = std::chrono::high_resolution_clock::now();
    auto scalar_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 测试SIMD版本
    start = std::chrono::high_resolution_clock::now();
    compute_distances_2d_simd(points, distances_simd);
    end = std::chrono::high_resolution_clock::now();
    auto simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 测试SIMD版本 8 float
    start = std::chrono::high_resolution_clock::now();
    compute_distances_2d_simd_8f(points, distances_simd_8f);
    end = std::chrono::high_resolution_clock::now();
    auto simd_8f_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    start = std::chrono::high_resolution_clock::now();
    compute_distances_2d_simd_8f(points, distances_simd_8f_ori);
    end = std::chrono::high_resolution_clock::now();
    auto simd_8f_ori_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 测试SOA SIMD版本
    start = std::chrono::high_resolution_clock::now();
    compute_distances_2d_soa_simd(points_soa, distances_soa);
    end = std::chrono::high_resolution_clock::now();
    auto soa_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 测试SOA SIMD版本 8 float
    start = std::chrono::high_resolution_clock::now();
    compute_distances_2d_soa_simd_8f(points_soa, distances_soa_8f);
    end = std::chrono::high_resolution_clock::now();
    auto soa_8f_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    start = std::chrono::high_resolution_clock::now();
    compute_distances_2d_soa_simd_8f(points_soa, distances_soa_8f_ori);
    end = std::chrono::high_resolution_clock::now();
    auto soa_8f_ori_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    start = std::chrono::high_resolution_clock::now();
    compute_distances_2d_soa_simd_8f(input_x, input_y, num_points, distances_soa_8f_ori_opt);
    end = std::chrono::high_resolution_clock::now();
    auto soa_8f_ori_opt_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 验证结果一致性
    bool results_match = true;
    for (size_t i = 0; i < distances_scalar.size() && i < 1000; ++i) {
        if (std::abs(distances_scalar[i] - distances_simd[i]) > 1e-5f) {
            results_match = false;
            break;
        }
        if (std::abs(distances_scalar[i] - distances_simd_8f[i]) > 1e-5f) {
            results_match = false;
            break;
        }
        if (std::abs(distances_scalar[i] - distances_simd_8f_ori[i]) > 1e-5f) {
            results_match = false;
            break;
        }
        if (std::abs(distances_scalar[i] - distances_soa[i]) > 1e-5f) {
            results_match = false;
            break;
        }
        if (std::abs(distances_scalar[i] - distances_soa_8f[i]) > 1e-5f) {
            results_match = false;
            break;
        }
        if (std::abs(distances_scalar[i] - distances_soa_8f_ori[i]) > 1e-5f) {
            results_match = false;
            break;
        }
        if (std::abs(distances_scalar[i] - distances_soa_8f_ori_opt[i]) > 1e-5f) {
            results_match = false;
            break;
        }
    }
    
    std::cout << "=== Performance Benchmark Results ===\n";
    std::cout << "Number of points: " << num_points << "\n";
    std::cout << "Scalar version:   " << scalar_time.count() << " μs\n";
    std::cout << "SIMD version:     " << simd_time.count() << " μs\n";
    std::cout << "SIMD 8f version:     " << simd_8f_time.count() << " μs\n";
    std::cout << "SIMD 8f ori version:     " << simd_8f_ori_time.count() << " μs\n";
    std::cout << "SOA SIMD version: " << soa_time.count() << " μs\n";
    std::cout << "SOA SIMD 8f version: " << soa_8f_time.count() << " μs\n";
    std::cout << "SOA SIMD 8f ori version: " << soa_8f_ori_time.count() << " μs\n";
    std::cout << "SOA SIMD 8f ori opt version: " << soa_8f_ori_opt_time.count() << " μs\n";
    std::cout << "SIMD speedup:     " << (double)scalar_time.count() / simd_time.count() << "x\n";
    std::cout << "SOA speedup:      " << (double)scalar_time.count() / soa_time.count() << "x\n";
    std::cout << "SIMD 8f speedup:     " << (double)scalar_time.count() / simd_8f_time.count() << "x\n";
    std::cout << "SOA 8f speedup:      " << (double)scalar_time.count() / soa_8f_time.count() << "x\n";
    std::cout << "SIMD 8f ori speedup:     " << (double)scalar_time.count() / simd_8f_ori_time.count() << "x\n";
    std::cout << "SOA 8f ori speedup:      " << (double)scalar_time.count() / soa_8f_ori_time.count() << "x\n";
    std::cout << "Results match:    " << (results_match ? "Yes" : "No") << "\n\n";

    free(distances_simd_8f_ori);
    free(distances_soa_8f_ori);
    free(distances_soa_8f_ori_opt);

    free(input_x);
    free(input_y);
}

int main() {
    std::cout << "SIMD Point Distance Calculation Demo\n";
    std::cout << "====================================\n\n";
    
    // 简单示例
    std::vector<Point2D> test_points = {
        {0, 0}, {3, 4}, {6, 8}, {9, 12}, {12, 16}
    };
    
    std::vector<float> distances_scalar, distances_simd;
    
    compute_distances_2d_scalar(test_points, distances_scalar);
    compute_distances_2d_simd(test_points, distances_simd);
    
    std::cout << "Test points and distances:\n";
    for (size_t i = 0; i < test_points.size(); ++i) {
        std::cout << "Point " << i << ": (" << test_points[i].x << ", " << test_points[i].y << ")\n";
        if (i < distances_scalar.size()) {
            std::cout << "  Distance to next: " << distances_scalar[i] 
                      << " (SIMD: " << distances_simd[i] << ")\n";
        }
    }
    std::cout << "\n";
    
    // 性能测试
    benchmark_distance_calculation();
    
    return 0;
}