sudo apt install doxygen graphviz clang-format clang-tidy

doxygen -g
doxygen Doxyfile


clang-format -i myfile.cpp
clang-tidy main.cpp

g++ main.cpp --std=c++20 -ftime-report -O3 -march=native -fsanitize=address

rm -rf build
cmake -B build -DCMAKE_SYSTEM_NAME=Android -DCMAKE_ANDROID_NDK=$ANDROID_NDK -DCMAKE_ANDROID_ARCH_ABI=arm64-v8a -DCMAKE_ANDROID_PLATFORM=24

https://arxiv.org/abs/1811.05031

sudo apt-get install libgtest-dev libgmock-dev