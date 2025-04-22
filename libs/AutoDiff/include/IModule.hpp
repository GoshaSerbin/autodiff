/**
 * @file IModule.hpp
 * @brief Defines the IModule interface and macros for creating modules in an automatic differentiation framework.
 *
 * This file provides the base interface `IModule` for modules in a computational graph
 * and a set of macros to define module concepts and implementations. These modules
 * are used to perform forward and backward passes for automatic differentiation.
 *
 * Key components:
 * - `IModule`: An abstract base class for modules, requiring a `Forward` method.
 * - `DEFINE_CONCEPT`: A macro to define a concept for a module backend with specific
 *   forward and backward function signatures.
 * - `DEFINE_CONCEPT_WITH_PARAMS`: A macro to define a concept for a module backend
 *   that requires additional parameters for forward and backward functions.
 * - `DEFINE_MODULE`: A macro to define a module class that implements the `IModule`
 *   interface using a backend that satisfies the defined concept.
 * - `DEFINE_MODULE_WITH_PARAMS`: A macro to define a module class that implements
 *   the `IModule` interface using a backend with additional parameters.
 *
 * These components allow for flexible and extensible module definitions, enabling
 * the creation of computational graphs with automatic differentiation capabilities.
 *
 * For most cases you should use predefined macros for your modules, however it is possible to implement Module class by your own if you
 * need custom features
 *
 * @note Examples of using this interface can be checked in tests.
 *
 * @warning It is `Backend` responsibility to verify  number of inputs and shapes of nodes' data. `Module` is only responsible for
 * graph creation and proper call of `Forward` and `Backward`
 *
 * @note In backend's Forward method `std::vector<NodePtr<T>>& outputs` is empty and should be filled during this method
 *
 * @note In backend's Backward method all original inputs are passed, even whose who doesn't require grad, so you should check
 * `requiresGrad` before computing grad
 */

#pragma once

#include <memory>
#include <type_traits>
#include <vector>

#include "Node.hpp"

namespace auto_diff {

    template <typename T>
    class IModule {
     public:
        [[nodiscard]] virtual auto Forward(const std::vector<NodePtr<T>>& inputs) -> std::vector<NodePtr<T>> = 0;
        virtual ~IModule() {}
    };

#define DEFINE_CONCEPT(ModuleName)                                                                                                     \
    template <typename Backend, typename T>                                                                                            \
    concept ModuleName##Backend = requires {                                                                                           \
        requires std::is_same_v<decltype(&Backend::ModuleName##Forward),                                                               \
                                void (*)(const std::vector<NodePtr<T>>&, std::vector<NodePtr<T>>&)>;                                   \
        requires std::is_same_v<decltype(&Backend::ModuleName##Backward), void (*)(const std::vector<NodePtr<T>>&, Node<T>*, size_t)>; \
    };

#define DEFINE_CONCEPT_WITH_PARAMS(ModuleName, ...)                                                               \
    template <typename Backend, typename T>                                                                       \
    concept ModuleName##Backend = requires {                                                                      \
        requires std::is_same_v<decltype(&Backend::ModuleName##Forward),                                          \
                                void (*)(const std::vector<NodePtr<T>>&, std::vector<NodePtr<T>>&, __VA_ARGS__)>; \
        requires std::is_same_v<decltype(&Backend::ModuleName##Backward),                                         \
                                void (*)(const std::vector<NodePtr<T>>&, Node<T>*, size_t, __VA_ARGS__)>;         \
    };

#define DEFINE_MODULE(ModuleName)                                                                                              \
    DEFINE_CONCEPT(ModuleName)                                                                                                 \
                                                                                                                               \
    template <typename T, ModuleName##Backend<T> Backend>                                                                      \
    class ModuleName : public IModule<T> {                                                                                     \
     public:                                                                                                                   \
        [[nodiscard]] auto Forward(const std::vector<NodePtr<T>>& inputs) -> std::vector<NodePtr<T>> override {                \
            std::vector<NodePtr<T>> outputs;                                                                                   \
            const bool requiresGrad = std::any_of(inputs.begin(), inputs.end(), [](NodePtr<T> t) { return t->requiresGrad; }); \
            Backend::ModuleName##Forward(inputs, outputs);                                                                     \
            for (size_t i = 0; i < outputs.size(); ++i) {                                                                      \
                auto output = outputs[i];                                                                                      \
                output->parents = inputs;                                                                                      \
                output->requiresGrad = requiresGrad;                                                                           \
                if (requiresGrad) {                                                                                            \
                    output->backwardFn = [inputs, out = output.get(), i]() { Backend::ModuleName##Backward(inputs, out, i); }; \
                }                                                                                                              \
            }                                                                                                                  \
            return outputs;                                                                                                    \
        }                                                                                                                      \
    };

#define DEFINE_MODULE_WITH_PARAMS(ModuleName, ...)                                                                             \
    DEFINE_CONCEPT_WITH_PARAMS(ModuleName, __VA_ARGS__)                                                                        \
                                                                                                                               \
    template <typename T, ModuleName##Backend<T> Backend>                                                                      \
    class ModuleName : public IModule<T> {                                                                                     \
     public:                                                                                                                   \
        explicit ModuleName(__VA_ARGS__ params) : m_params(params) {}                                                          \
        [[nodiscard]] auto Forward(const std::vector<NodePtr<T>>& inputs) -> std::vector<NodePtr<T>> override {                \
            std::vector<NodePtr<T>> outputs;                                                                                   \
            const bool requiresGrad = std::any_of(inputs.begin(), inputs.end(), [](NodePtr<T> t) { return t->requiresGrad; }); \
            Backend::ModuleName##Forward(inputs, outputs, m_params);                                                           \
            for (size_t i = 0; i < outputs.size(); ++i) {                                                                      \
                auto output = outputs[i];                                                                                      \
                output->parents = inputs;                                                                                      \
                output->requiresGrad = requiresGrad;                                                                           \
                if (requiresGrad) {                                                                                            \
                    output->backwardFn = [inputs, out = output.get(), i, params = m_params]() {                                \
                        Backend::ModuleName##Backward(inputs, out, i, params);                                                 \
                    };                                                                                                         \
                }                                                                                                              \
            }                                                                                                                  \
            return outputs;                                                                                                    \
        }                                                                                                                      \
                                                                                                                               \
     private:                                                                                                                  \
        __VA_ARGS__ m_params;                                                                                                  \
    };

}  // namespace auto_diff