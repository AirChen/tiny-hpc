#include <algorithm>
#include <cstdio>
#include <iterator>
#include <sys/types.h>
#include <xmmintrin.h>
#pragma GCC target("avx2")
#pragma GCC optimize("O3")

#include <x86intrin.h>
#include <bits/stdc++.h>
#include <iostream>
using namespace std;

typedef __m128 Packet4f;
template <int p, int q, int r, int s>
struct shuffle_mask {
  enum { mask = (s) << 6 | (r) << 4 | (q) << 2 | (p) };
};

// TODO: change the implementation of all swizzle* ops from macro to template,
#define vec4f_swizzle1(v, p, q, r, s) \
  Packet4f(_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v), (shuffle_mask<p, q, r, s>::mask))))

#define vec4f_swizzle2(a, b, p, q, r, s) Packet4f(_mm_shuffle_ps((a), (b), (shuffle_mask<p, q, r, s>::mask)))

Packet4f vec4f_movelh(const Packet4f& a, const Packet4f& b) {
  return Packet4f(_mm_movelh_ps(a, b));
}
Packet4f vec4f_movehl(const Packet4f& a, const Packet4f& b) {
  return Packet4f(_mm_movehl_ps(a, b));
}
/*
a = [A1, A2, A3, A4]
b = [B1, B2, B3, B4]
result = [A1, B1, A2, B2]  // 低位交错
*/
Packet4f vec4f_unpacklo(const Packet4f& a, const Packet4f& b) {
  return Packet4f(_mm_unpacklo_ps(a, b));
}
/*
a = [A1, A2, A3, A4]
b = [B1, B2, B3, B4]
result = [A3, B3, A4, B4]  // 高位交错
*/
Packet4f vec4f_unpackhi(const Packet4f& a, const Packet4f& b) {
  return Packet4f(_mm_unpackhi_ps(a, b));
}
// 将指定位置的元素广播到整个向量
#define vec4f_duplane(a, p) vec4f_swizzle2(a, a, p, p, p, p)

void print_vec4f(const Packet4f& t) {
  float ft[4];
  _mm_store_ps(&ft[0], t);
  std::cout << "ans: " << ft[0] << ", " << ft[1] << ", " << ft[2] << ", " << ft[3] << std::endl;
}

void print_floats(const float* t) {
  std::cout << "ans: " << t[0] << ", " << t[1] << ", " << t[2] << ", " << t[3] << std::endl;
}

Packet4f load_packet(const float* t) {
  return _mm_load_ps(t);
}

void store_packet(float* out, const Packet4f& t) {
  _mm_store_ps(out, t);
}

Packet4f pset1(const float& from) {
  return _mm_set_ps1(from);
}

// 奇数位mask
Packet4f peven_mask() {
  return _mm_castsi128_ps(_mm_set_epi32(0, -1, 0, -1));
}

// 偶数位mask
Packet4f podd_mask() {
  return _mm_castsi128_ps(_mm_set_epi32(-1, 0, -1, 0));
}

// 取反
Packet4f pnegate(const Packet4f& a) {
  const Packet4f mask = _mm_castsi128_ps(_mm_setr_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000));
  return _mm_xor_ps(a, mask);
}

__m128 abs_ps(__m128 x) {
    // 符号位掩码：只有最高位为1
    const __m128 sign_mask = _mm_set1_ps(-0.0f);  // 0x80000000
    
    // 清除符号位
    return _mm_andnot_ps(sign_mask, x); // (~a) & b
}

Packet4f pabs(const Packet4f& a) {
  const __m128i mask = _mm_setr_epi32(0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF);
  return _mm_castsi128_ps(_mm_and_si128(mask, _mm_castps_si128(a)));
}

Packet4f preverse(const Packet4f& a) {
  return _mm_shuffle_ps(a, a, 0x1B);
}

float pfirst(const Packet4f& a) {
  return _mm_cvtss_f32(a);
}

Packet4f pgather(const float* from, uint stride) {
  return _mm_set_ps(from[3 * stride], from[2 * stride], from[1 * stride], from[0 * stride]);
}

void pscatter(float* to, const Packet4f& from, uint stride) {
  to[stride * 0] = pfirst(from);
  to[stride * 1] = pfirst(_mm_shuffle_ps(from, from, 1));
  to[stride * 2] = pfirst(_mm_shuffle_ps(from, from, 2));
  to[stride * 3] = pfirst(_mm_shuffle_ps(from, from, 3));

  // _mm_shuffle_ps(from, from, 1) 的作用：
  // 将向量中的第1个元素移动到第0位
  // pfirst() 提取第0位的元素
  // 相当于提取原向量的第1个元素
}

// template <>
// EIGEN_STRONG_INLINE Packet4f pcmp_le(const Packet4f& a, const Packet4f& b) {
//   return _mm_cmple_ps(a, b); // a <= b
// }
// template <>
// EIGEN_STRONG_INLINE Packet4f pcmp_lt(const Packet4f& a, const Packet4f& b) {
//   return _mm_cmplt_ps(a, b); // a < b
// }
// template <>
// EIGEN_STRONG_INLINE Packet4f pcmp_lt_or_nan(const Packet4f& a, const Packet4f& b) {
//   return _mm_cmpnge_ps(a, b);
// }
// template <>
// EIGEN_STRONG_INLINE Packet4f pcmp_eq(const Packet4f& a, const Packet4f& b) {
//   return _mm_cmpeq_ps(a, b); // a == b
// }

// template <>
// EIGEN_STRONG_INLINE Packet4f pcmp_gt(const Packet4f& a, const Packet4f& b) {
//   return _mm_cmpgt_ps(a, b);  // a > b
// }

// template <>
// EIGEN_STRONG_INLINE Packet4f pcmp_ge(const Packet4f& a, const Packet4f& b) {
//   return _mm_cmpge_ps(a, b);  // a >= b
// }

// template <>
// EIGEN_STRONG_INLINE Packet4f pcmp_ne(const Packet4f& a, const Packet4f& b) {
//   return _mm_cmpneq_ps(a, b);  // a != b
// }

void transpose_4x4(const float matrix[][4], float* x, float* y, float* z, float* w) {
    Packet4f row0 = load_packet(matrix[0]);  // [a00, a01, a02, a03]
    Packet4f row1 = load_packet(matrix[1]);  // [a10, a11, a12, a13]
    Packet4f row2 = load_packet(matrix[2]);  // [a20, a21, a22, a23]
    Packet4f row3 = load_packet(matrix[3]);  // [a30, a31, a32, a33]
    
    // 使用unpack指令进行转置
    Packet4f tmp0 = vec4f_unpacklo(row0, row1);  // [a00, a10, a01, a11]
    Packet4f tmp1 = vec4f_unpacklo(row2, row3);  // [a20, a30, a21, a31]
    Packet4f tmp2 = vec4f_unpackhi(row0, row1);  // [a02, a12, a03, a13]
    Packet4f tmp3 = vec4f_unpackhi(row2, row3);  // [a22, a32, a23, a33]
    
    row0 = vec4f_movelh(tmp0, tmp1);  // [a00, a10, a20, a30]
    row1 = vec4f_movehl(tmp0, tmp1);  // [a01, a11, a21, a31]
    row2 = vec4f_movelh(tmp2, tmp3);  // [a02, a12, a22, a32]
    row3 = vec4f_movehl(tmp2, tmp3);  // [a03, a13, a23, a33]

    store_packet(x, row0);
    store_packet(y, row1);
    store_packet(z, row2);
    store_packet(w, row3);
}

int main() {
  // number of registers
  // 32 bits =>  8 registers
  // 64 bits => 16 registers
  std::cout << 2 * sizeof(void *) << std::endl;

  // shuffle
  float a[4];
  {
    a[0] = 23.0;
    a[1] = 0.0;
    a[2] = 12.5;
    a[3] = 1304.0;
  }
  __m128 x = _mm_load_ps(&a[0]);
  __m128 y = vec4f_swizzle1(x, 3, 2, 1, 0);
  __m128 z = vec4f_swizzle2(x, y, 0, 1, 0, 1);// 取x 的前两位 取y 的后两位

  std::cout << "input: " << a[0] << ", " << a[1] << ", " << a[2] << ", " << a[3] << std::endl;
  print_vec4f(y);
  print_vec4f(z);

  __m128 t1 = vec4f_movelh(x, y); // 取a的低64位 + b的低64位
  print_vec4f(t1);
  
  __m128 t2 = vec4f_movehl(x, y); // 取a的高64位 + b的高64位
  print_vec4f(t2);
  
  __m128 t3 = vec4f_unpackhi(x, y);
  print_vec4f(t3);
  
  __m128 t4 = vec4f_unpacklo(x, y);
  print_vec4f(t4);

  __m128 t5 = vec4f_duplane(x, 3);
  print_vec4f(t5);

  auto print_matrix = [](const float t[][4]) {
    for (int i = 0; i < 4; i++) {
      print_floats(t[i]);
    }
  };

  float ma[4][4];
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      ma[i][j] = j + 1;
    }
  }
  float mb[4][4];
  transpose_4x4(ma, mb[0], mb[1], mb[2], mb[3]);

  std::cout << "input matrix: \n";
  print_matrix(ma);
  std::cout << "output matrix: \n";
  print_matrix(mb);

  __m128 t6 = pset1(23);
  print_vec4f(t6);

  __m128 tmp = peven_mask();
  __m128 t7 = _mm_add_ps(t6, tmp);
  print_vec4f(t7);

  tmp = podd_mask();
  __m128 t8 = _mm_add_ps(t6, tmp);
  print_vec4f(t8);

  __m128 t9 = _mm_add_ss(t7, t8);
  print_vec4f(t9);
  
  __m128 t10 = _mm_add_ss(t8, t7);
  print_vec4f(t10);

  __m128 t11 = _mm_add_ps(t7, t8);
  print_vec4f(t11);

  __m128 ta = _mm_set_ps(8.0f, 6.0f, 4.0f, 2.0f);  // [2, 4, 6, 8]
  __m128 tb = _mm_set_ps(1.0f, 1.0f, 1.0f, 1.0f);  // [1, 1, 1, 1]
  
  __m128 result = _mm_addsub_ps(ta, tb);// 1 5 5 9
  print_vec4f(result);

  ta = _mm_set_ps(-1, -12, 123, -144);
  ta = vec4f_swizzle1(ta, 3, 2, 1, 0);
  __m128 t12 = pnegate(ta);
  print_vec4f(t12);

  __m128 t13 = abs_ps(ta);
  print_vec4f(t13);

  __m128 t14 = pabs(ta);
  print_vec4f(t14);

  __m128 t15 = preverse(t14);
  print_vec4f(t15);

  std::cout << "t14 first: " << pfirst(t14) << " t15 first: " << pfirst(t15) << std::endl;

  return 0;
}