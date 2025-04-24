// #include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cmath>

#include "IModule.hpp"
#include "Node.hpp"

using auto_diff::Node;
using auto_diff::NodePtr;

template <typename T>
class Vec : public std::vector<T> {
 public:
    using std::vector<T>::vector;
    auto operator=(int value) -> Vec& {
        for (auto& elem : *this) {
            elem = value;
        }
        return *this;
    }
};

template <typename T>
using VecNode = Node<Vec<T>>;

template <typename T>
using VecNodePtr = NodePtr<Vec<T>>;

template <typename T>
class VectorBackend {
 public:
    static auto VectorSumForward(const std::vector<VecNodePtr<T>>& inputs, std::vector<VecNodePtr<T>>& outputs) -> void {
        outputs.push_back(std::make_shared<VecNode<T>>(Vec<T>(inputs[0]->data.size(), 0)));
        for (const auto& input : inputs) {
            for (size_t i = 0; i < input->data.size(); ++i) {
                outputs[0]->data[i] += input->data[i];
            }
        }
    }
    static auto VectorSumBackward(const std::vector<VecNodePtr<T>>& inputs, VecNode<T>* output, [[maybe_unused]] size_t outputIdx) -> void {
        for (const auto& input : inputs) {
            if (input->requiresGrad) {
                for (size_t i = 0; i < input->data.size(); ++i) {
                    input->grad[i] += output->grad[i];
                }
            }
        }
    }

    static auto VectorSplitForward(const std::vector<VecNodePtr<T>>& inputs, std::vector<VecNodePtr<T>>& outputs) -> void {
        const auto& input = inputs[0];
        const size_t size = input->data.size();
        for (size_t i = 0; i < size; ++i) {
            outputs.push_back(std::make_shared<VecNode<T>>(Vec<T>(1, input->data[i])));
        }
    }
    static auto VectorSplitBackward(const std::vector<VecNodePtr<T>>& inputs, VecNode<T>* output, size_t outputIdx) -> void {
        const auto& input = inputs[0];
        if (input->requiresGrad) {
            input->grad[outputIdx] += output->grad[0];
        }
    }
    static auto VectorPowForward(const std::vector<VecNodePtr<T>>& inputs, std::vector<VecNodePtr<T>>& outputs, T pow) -> void {
        const auto& input = inputs[0];
        outputs.push_back(std::make_shared<VecNode<T>>(Vec<T>(input->data.size(), 0)));
        for (size_t i = 0; i < input->data.size(); ++i) {
            outputs[0]->data[i] = std::pow(input->data[i], pow);
        }
    }
    static auto VectorPowBackward(const std::vector<VecNodePtr<T>>& inputs,
                                  VecNode<T>* output,
                                  [[maybe_unused]] size_t outputIdx,
                                  T pow) -> void {
        const auto& input = inputs[0];
        if (input->requiresGrad) {
            for (size_t i = 0; i < input->data.size(); ++i) {
                input->grad[i] += pow * std::pow(input->data[i], pow - 1) * output->grad[i];
            }
        }
    }
};

namespace auto_diff {
    DEFINE_MODULE(VectorSum)
}

TEST(AutoDiffTest, Sum2VectorsCase) {
    auto_diff::VectorSum<Vec<int>, VectorBackend<int>> module;
    auto a = std::make_shared<VecNode<int>>(Vec<int>{1, 2, 3, 4}, true);
    auto b = std::make_shared<VecNode<int>>(Vec<int>{1, 2, 3, 4}, true);
    auto c = module.Forward({a, b})[0];
    EXPECT_EQ(c->data, Vec<int>({2, 4, 6, 8}));
    c->Backward();
    EXPECT_EQ(a->grad, Vec<int>({1, 1, 1, 1}));
    EXPECT_EQ(b->grad, Vec<int>({1, 1, 1, 1}));
}
namespace auto_diff {
    DEFINE_MODULE(VectorSplit)
}

TEST(AutoDiffTest, SplitVectorCase) {
    auto_diff::VectorSplit<Vec<int>, VectorBackend<int>> module;
    auto a = std::make_shared<VecNode<int>>(Vec<int>{1, 2, 3, 4}, true);
    auto parts = module.Forward({a});
    EXPECT_EQ(parts.size(), 4);
    EXPECT_EQ(parts[0]->data, Vec<int>({1}));
    EXPECT_EQ(parts[1]->data, Vec<int>({2}));
    EXPECT_EQ(parts[2]->data, Vec<int>({3}));
    EXPECT_EQ(parts[3]->data, Vec<int>({4}));
    parts[2]->Backward();
    EXPECT_EQ(a->grad, Vec<int>({0, 0, 1, 0}));
}

namespace auto_diff {
    DEFINE_MODULE_WITH_PARAMS(VectorPow, int)
}

TEST(AutoDiffTest, ModuleWithParamsCase) {
    auto_diff::VectorPow<Vec<int>, VectorBackend<int>> module(2);
    auto a = std::make_shared<VecNode<int>>(Vec<int>{1, 2, 3, 4}, true);
    auto b = module.Forward({a})[0];
    EXPECT_EQ(b->data, Vec<int>({1, 4, 9, 16}));
    b->Backward();
    EXPECT_EQ(a->grad, Vec<int>({2, 4, 6, 8}));
}