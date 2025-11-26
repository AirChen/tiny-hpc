#include <iostream>

void test_stag0(int ths) {
    for (int i = 0; i < ths; i++) {
        std::cout << "-> " << i << " ";
        for (int c = 1; c < ths; c *= 2) {
            if (i % (2 * c) == 0) {
                std::cout << (i + c) << " ";
            }
        }
        std::cout << "\n";
    }
}

void test_stag1(int ths) {
    for (int i = 0; i < ths; i++) {
        std::cout << "-> " << i << ": ";
        for (int c = 1; c < ths; c *= 2) {
            int index = i * 2 * c;
            if (index < ths) {
                std::cout << "<" << index << " + " << (index + c) << "> ";
            }
        }
        std::cout << "\n";
    } 
}

void test_stag2(int ths) {
    for (int i = 0; i < ths; i++) {
        std::cout << "-> " << i << ": ";
        for (int c = ths / 2; c > 0; c >>= 1) {
            if (i < c) {
                std::cout << " + " << (i + c) << " ";
            }
        }
        std::cout << "\n";
    } 
}

int main() {
    int ths = 32;
    test_stag0(ths);
    test_stag1(ths);
    test_stag2(ths);
    return 0;
}
