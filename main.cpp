#ifdef USE_OPENMP
#include <omp.h>
#endif

#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <memory>

// #include <_Float16.hpp>
#include <string>
#include <unordered_set>

#ifdef ANDROID
using float16_t = _Float16;
#else
#include <stdfloat>
#if __STDCPP_FLOAT64_T__ == 1
using float16_t = std::float16_t;
#else
#error "16-bit float type support required"
#endif
#endif

#include "Logger.hpp"
#include "Stopwatch.hpp"

template <typename DTYPE>
class TensorData {
 public:
    TensorData(std::vector<size_t> shape) : shape(shape) {
        size = 1;
        for (size_t dim : shape) size *= dim;
        data = new DTYPE[size]();
    }

    TensorData(const TensorData& other) : shape(other.shape), size(other.size), data(new DTYPE[other.size]) {
        std::copy(other.data, other.data + size, data);
    }

    TensorData(TensorData&& other) noexcept : shape(std::move(other.shape)), size(other.size), data(other.data) {
        other.data = nullptr;
        other.size = 0;
    }

    ~TensorData() { delete[] data; }

    TensorData& operator=(const TensorData& other) {
        if (this != &other) {
            delete[] data;
            shape = other.shape;
            size = other.size;
            data = new DTYPE[size];
            std::copy(other.data, other.data + size, data);
        }
        return *this;
    }

    TensorData& operator=(TensorData&& other) noexcept {
        if (this != &other) {
            delete[] data;
            shape = std::move(other.shape);
            size = other.size;
            data = other.data;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }

    TensorData& operator=(DTYPE value) {
        for (size_t i = 0; i < size; ++i) {
            data[i] = value;
        }
        return *this;
    }

    DTYPE& operator[](std::vector<size_t> indices) { return data[get_flat_index(indices)]; }

    const DTYPE& operator[](std::vector<size_t> indices) const { return data[get_flat_index(indices)]; }

    TensorData& operator+=(const TensorData& other) {
        check_size(other);
        for (size_t i = 0; i < size; ++i) {
            data[i] += other.data[i];
        }
        return *this;
    }

    TensorData& operator-=(const TensorData& other) {
        check_size(other);
        for (size_t i = 0; i < size; ++i) {
            data[i] -= other.data[i];
        }
        return *this;
    }

    TensorData operator+(const TensorData& other) const {
        TensorData result = *this;
        result += other;
        return result;
    }

    TensorData operator-(const TensorData& other) const {
        TensorData result = *this;
        result -= other;
        return result;
    }

    TensorData operator*(const TensorData& other) const {
        check_size(other);
        TensorData result(shape);
        for (size_t i = 0; i < size; ++i) {
            result.data[i] = data[i] * other.data[i];
        }
        return result;
    }

    TensorData operator/(const TensorData& other) const {
        check_size(other);
        TensorData result(shape);
        for (size_t i = 0; i < size; ++i) {
            if (other.data[i] == 0) throw std::runtime_error("Division by zero");
            result.data[i] = data[i] / other.data[i];
        }
        return result;
    }

    TensorData operator+(DTYPE value) const {
        TensorData result(shape);
        for (size_t i = 0; i < size; ++i) {
            result.data[i] = data[i] + value;
        }
        return result;
    }

    TensorData operator/(DTYPE value) const {
        if (value == 0) throw std::runtime_error("Division by zero");
        TensorData result(shape);
        for (size_t i = 0; i < size; ++i) {
            result.data[i] = data[i] / value;
        }
        return result;
    }

    friend TensorData operator/(DTYPE value, const TensorData& tensor) {
        TensorData result(tensor.shape);
        for (size_t i = 0; i < tensor.size; ++i) {
            if (tensor.data[i] == 0) throw std::runtime_error("Division by zero");
            result.data[i] = value / tensor.data[i];
        }
        return result;
    }

    TensorData operator-() const {
        TensorData result(shape);
        for (size_t i = 0; i < size; ++i) {
            result.data[i] = -data[i];
        }
        return result;
    }

    size_t get_size() const { return size; }

    friend std::ostream& operator<<(std::ostream& os, const TensorData& tensor) {
        for (size_t i = 0; i < tensor.size; ++i) {
            os << tensor.data[i] << " ";
        }
        return os;
    }

 private:
    std::vector<size_t> shape;
    size_t size;
    DTYPE* data;

    size_t get_flat_index(const std::vector<size_t>& indices) const {
        if (indices.size() != shape.size()) {
            throw std::out_of_range("Incorrect number of indices");
        }
        size_t index = 0;
        size_t multiplier = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            if (indices[i] >= shape[i]) {
                throw std::out_of_range("Index out of bounds");
            }
            index += indices[i] * multiplier;
            multiplier *= shape[i];
        }
        return index;
    }

    void check_size(const TensorData& other) const {
        if (shape != other.shape) {
            throw std::invalid_argument("Tensor shapes do not match");
        }
    }
};

template <typename DATA>
class Tensor {
 public:
    DATA value;
    DATA grad;
    bool requires_grad;
    std::vector<std::shared_ptr<Tensor>> parents;
    std::function<void()> backward_fn;

    Tensor(DATA value, bool requires_grad = false) : value(value), grad(value), requires_grad(requires_grad) { grad = 0.; }

    void backward();
};

template <typename DATA>
std::shared_ptr<Tensor<DATA>> operator+(const std::shared_ptr<Tensor<DATA>>& a, const std::shared_ptr<Tensor<DATA>>& b) {
    auto out = std::make_shared<Tensor<DATA>>(a->value + b->value, a->requires_grad || b->requires_grad);
    if (out->requires_grad) {
        out->parents = {a, b};
        out->backward_fn = [a = a.get(), b = b.get(), out = out.get()]() {
            if (a->requires_grad) a->grad += out->grad;
            if (b->requires_grad) b->grad += out->grad;
        };
    }
    return out;
}

template <typename DATA>
std::shared_ptr<Tensor<DATA>> operator-(const std::shared_ptr<Tensor<DATA>>& a, const std::shared_ptr<Tensor<DATA>>& b) {
    auto out = std::make_shared<Tensor<DATA>>(a->value - b->value, a->requires_grad || b->requires_grad);
    if (out->requires_grad) {
        out->parents = {a, b};
        out->backward_fn = [a = a.get(), b = b.get(), out = out.get()]() {
            if (a->requires_grad) a->grad += out->grad;
            if (b->requires_grad) b->grad -= out->grad;
        };
    }
    return out;
}

template <typename DATA>
std::shared_ptr<Tensor<DATA>> operator*(const std::shared_ptr<Tensor<DATA>>& a, const std::shared_ptr<Tensor<DATA>>& b) {
    auto out = std::make_shared<Tensor<DATA>>(a->value * b->value, a->requires_grad || b->requires_grad);
    if (out->requires_grad) {
        out->parents = {a, b};
        out->backward_fn = [a = a.get(), b = b.get(), out = out.get()]() {
            if (a->requires_grad) a->grad += b->value * out->grad;
            if (b->requires_grad) b->grad += a->value * out->grad;
        };
    }
    return out;
}

template <typename DATA>
std::shared_ptr<Tensor<DATA>> operator/(const std::shared_ptr<Tensor<DATA>>& a, const std::shared_ptr<Tensor<DATA>>& b) {
    auto out = std::make_shared<Tensor<DATA>>(a->value / b->value, a->requires_grad || b->requires_grad);
    if (out->requires_grad) {
        out->parents = {a, b};
        out->backward_fn = [a = a.get(), b = b.get(), out = out.get()]() {
            if (a->requires_grad) a->grad += (1.0 / b->value) * out->grad;
            if (b->requires_grad) b->grad -= (a->value / (b->value * b->value)) * out->grad;
        };
    }
    return out;
}

// template <typename DATA>
// std::shared_ptr<Tensor<DATA>> pow(const std::shared_ptr<Tensor<DATA>>& a, double exponent) {
//     auto out = std::make_shared<Tensor<DATA>>(std::pow(a->value, exponent), a->requires_grad);
//     if (out->requires_grad) {
//         out->parents = {a};
//         out->backward_fn = [a = a.get(), exponent, out = out.get()]() {
//             if (a->requires_grad) a->grad += (exponent * std::pow(a->value, exponent - 1)) * out->grad;
//         };
//     }
//     return out;
// }

template <typename DATA>
std::shared_ptr<Tensor<DATA>> operator-(const std::shared_ptr<Tensor<DATA>>& a) {
    auto out = std::make_shared<Tensor<DATA>>(-(a->value), a->requires_grad);
    if (out->requires_grad) {
        out->parents = {a};
        out->backward_fn = [a = a.get(), out = out.get()]() {
            if (a->requires_grad) a->grad -= out->grad;
        };
    }
    return out;
}

template <typename DATA>
void Tensor<DATA>::backward() {
    if (!requires_grad) return;

    grad = 1.0;
    std::vector<Tensor*> topo_order;
    std::unordered_set<Tensor*> visited;

    std::function<void(Tensor*)> build_topo = [&](Tensor* node) {
        if (visited.count(node) || !node->requires_grad) return;
        visited.insert(node);
        for (auto& parent : node->parents) build_topo(parent.get());
        topo_order.push_back(node);
    };

    build_topo(this);

    for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
        if ((*it)->backward_fn) (*it)->backward_fn();
    }
}

auto main() -> int {
    // auto x = std::make_shared<Tensor<TensorData<float> > >(TensorData<float>({1}), true);
    // x->value = 1.f;
    // // float16_t r = 0;
    // LOG_WARNING("{} {}", "hi", PROJECT_NAME);
    // auto y = std::make_shared<Tensor<TensorData<float> > >(TensorData<float>({1}), true);
    // y->value[{0}] = 30.f;
    auto x = std::make_shared<Tensor<float>>(3.f, true);
    auto y = std::make_shared<Tensor<float>>(3.f, true);
    // y->value = 30.f;

    auto z = x + x + x + y;  //+ pow(x, 3);  // z = x * y + y
    // auto z2 = z * z;

    z->backward();

    std::cout << "dz/dx = " << x->grad << std::endl;  // Ожидаем 3 (производная по x)
    std::cout << "dz/dy = " << y->grad << std::endl;  // Ожидаем 3 (от умножения) + 1 (от сложения) = 4

    // auto file = std::make_unique<std::ofstream>("log.log");
    // Logger::GetInstance().SetOutputStream(file.get());
    // LOG_INFO("Let me log str {} here!", "hi");
    // const size_t N = 50;
    // auto file = std::make_unique<std::ifstream>("story.txt");
    // std::string s{};
    // while (std::getline(*file, s)) {  // пока не достигнут конец файла класть очередную строку в переменную (s)
    //     std::cout << s << std::endl;  // выводим на экран
    //     s += "+";                     // что нибудь делаем со строкой например я добавляю плюсик в конце каждой строки
    // }
    // Stopwatch watch;
    // float* A = new float[N * N](0.f);
    // float* B = new float[N * N](0.f);
    // float* C = new float[N * N](0.f);
    // for (size_t iter = 0; iter < 3; ++iter) {
    //     watch.Start();
    //     for (size_t i = 0; i < N; ++i) {
    //         for (size_t j = 0; j < N; ++j) {
    //             for (size_t k = 0; k < N; ++k) {
    //                 C[i * N + j] += A[i * N + k] * B[k * N + i];
    //             }
    //         }
    //     }
    //     watch.Stop();
    // }
    // std::cout << watch.GetAVG() << " " << watch.GetSTD() << std::endl;
    // watch.Reset();

    // for (size_t iter = 0; iter < 3; ++iter) {
    //     watch.Start();
    //     for (int i = 0; i < N; ++i) {
    //         float* c = C + i * N;
    //         for (int j = 0; j < N; ++j) c[j] = 0;
    //         for (int k = 0; k < N; ++k) {
    //             const float* b = B + k * N;
    //             float a = A[i * N + k];
    //             for (int j = 0; j < N; ++j) c[j] += a * b[j];
    //         }
    //     }
    //     watch.Stop();
    // }
    // std::cout << watch.GetAVG() << " " << watch.GetSTD() << std::endl;
    // watch.Reset();
    // delete[] A;
    // delete[] B;
    // delete[] C;

    // LOG_INFO("Got value {:f16}", 3.2);
    return 0;
}
