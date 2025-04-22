#include <exception>
#include <iostream>
#include <new>

#include "Logger.hpp"

// using shape_t = std::vector<size_t>;

// // ITensor for extensibility?
// template <typename T = float>
// struct Tensor {
//  public:
//     bool requiresGrad{true};
//     auto Shape() -> shape_t { return m_shape; }
//     auto Size() const noexcept -> size_t { return m_data.size(); }
//     Tensor(shape_t shape, bool requiresGrad = true) : requiresGrad(requiresGrad) {
//         m_shape = std::move(shape);
//         size_t n = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
//         m_data.resize(n, 0);
//         if (requiresGrad) {
//             m_grad.resize(m_data.size(), T{});
//         }
//     }
//     Tensor(std::vector<T> data, shape_t shape, bool requiresGrad = true) : requiresGrad(requiresGrad) {
//         size_t n = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
//         if (n != data.size()) {
//             throw std::exception();
//         }
//         m_data = std::move(data);
//         m_shape = std::move(shape);
//         if (requiresGrad) {
//             m_grad.resize(m_data.size(), T{});
//         }
//     }
//     auto SetData(std::vector<T> data) -> void {
//         if (data.size() != m_data.size()) {
//             throw std::exception();
//         }
//         m_data = std::move(data);
//         ZeroGrad();
//     }
//     auto Reshape(shape_t shape) -> void {
//         size_t n = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
//         if (n != m_data.size()) {
//             throw std::exception();
//         }
//         m_shape = std::move(shape);
//     }

//     auto ZeroGrad() -> void {
//         if (requiresGrad) {
//             std::fill(m_grad.begin(), m_grad.end(), T{});
//         }
//     }

//     // Неконстантный доступ
//     T& Data(std::size_t index) { return m_data[index]; }

//     // Константный доступ
//     const T& Data(std::size_t index) const { return m_data[index]; }
//     // Неконстантный доступ
//     T& Grad(std::size_t index) { return m_grad[index]; }

//     // Константный доступ
//     const T& Grad(std::size_t index) const { return m_grad[index]; }

//     std::function<void()> m_backwardFn;

//     void Backward() {
//         if (!requiresGrad) return;

//         for (size_t i = 0; i < m_grad.size(); ++i) {
//             std::cout << m_grad.size() << std::endl;
//             m_grad[i] = 1;
//         }

//         if (m_backwardFn) {
//             m_backwardFn();
//         }
//         // std::vector<Tensor*> topo_order;
//         // std::unordered_set<Variable*> visited;

//         // std::function<void(Tensor*)> build_topo = [&](Tensor* node) {
//         //     if (visited.count(node) || !node->requiresGrad) return;
//         //     visited.insert(node);
//         //     for (auto& parent : node->parents) build_topo(parent.get());
//         //     topo_order.push_back(node);
//         // };

//         // build_topo(this);

//         // for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
//         //     if ((*it)->backwardFn) (*it)->backwardFn();
//         // }
//     }

//  private:
//     std::vector<T> m_data;
//     std::vector<size_t> m_shape;
//     std::vector<T> m_grad;
// };

auto main() -> int {
    try {
        LOG_INFO("hi, how you are doing?");
    } catch (const std::bad_alloc& e) {
        std::cerr << "Ошибка: нехватка памяти: " << e.what() << '\n';
    } catch (const std::ios_base::failure& e) {
        std::cerr << "Ошибка ввода/вывода: " << e.what() << '\n';
    } catch (const std::exception& e) {
        std::cerr << "Другая ошибка: " << e.what() << '\n';
    }
    return 0;
}
