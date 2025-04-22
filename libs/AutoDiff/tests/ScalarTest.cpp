// #include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "Modules.hpp"

using namespace auto_diff;

TEST(AutoDiffTest, NodeCreationCase) {
    auto a = std::make_shared<Node<int>>();
    EXPECT_EQ(a->data, 0);
    EXPECT_EQ(a->grad, 0);
    EXPECT_EQ(a->requiresGrad, true);
    EXPECT_EQ(a->parents.size(), 0);
    EXPECT_EQ(a->backwardFn, nullptr);
}

// нельзя называть класс Backend'а так же как и имя модуля

template <typename T>
class BasicBackend {
 public:
    static auto BasicSumForward(const std::vector<NodePtr<T>>& inputs, std::vector<NodePtr<T>>& outputs) -> void {
        outputs.push_back(std::make_shared<Node<T>>());
        for (auto& input : inputs) {
            outputs[0]->data += input->data;
        }
    }
    static auto BasicSumBackward(const std::vector<NodePtr<T>>& inputs, Node<T>* output, [[maybe_unused]] size_t outputIdx) -> void {
        for (auto& input : inputs) {
            if (input->requiresGrad) {
                input->grad += output->grad;
            }
        }
    }
};

DEFINE_MODULE(BasicSum)

TEST(AutoDiffTest, ForwardCaseSum2Numbers) {
    BasicSum<int, BasicBackend<int>> module;
    auto a = std::make_shared<Node<int>>(3);
    auto b = std::make_shared<Node<int>>(4);
    auto c = module.Forward({a, b})[0];
    EXPECT_EQ(c->data, 7);
    EXPECT_EQ(c->grad, 0);
    EXPECT_EQ(c->parents.size(), 2);
    EXPECT_EQ(a->parents.size(), 0);
    EXPECT_EQ(b->parents.size(), 0);
    EXPECT_FALSE(a->backwardFn);
    EXPECT_FALSE(b->backwardFn);
    EXPECT_TRUE(c->backwardFn);
}

TEST(AutoDiffTest, BackwarCaseSum2Numbers) {
    BasicSum<int, BasicBackend<int>> module;
    auto a = std::make_shared<Node<int>>(3);
    auto b = std::make_shared<Node<int>>(4);
    auto c = module.Forward({a, b})[0];
    c->Backward();
    EXPECT_EQ(c->grad, 1);
    EXPECT_EQ(a->grad, 1);
    EXPECT_EQ(b->grad, 1);
}

TEST(AutoDiffTest, SumManyInputsCase) {
    BasicSum<int, BasicBackend<int>> module;
    std::vector<NodePtr<int>> inputs;
    const size_t n = 10;
    for (size_t i = 1; i <= n; ++i) {
        inputs.push_back(std::make_shared<Node<int>>(i));
    }
    auto c = module.Forward(inputs)[0];
    c->Backward();
    EXPECT_EQ(c->data, n * (n + 1) / 2);
    EXPECT_EQ(c->grad, 1);
    for (const auto& input : inputs) {
        EXPECT_EQ(input->grad, 1);
    }
}

TEST(AutoDiffTest, SequantialCallOfModuleCase) {
    BasicSum<int, BasicBackend<int>> module;
    auto a = std::make_shared<Node<int>>(10);
    auto b = std::make_shared<Node<int>>(100);
    auto c = module.Forward({a, b})[0];
    auto d = module.Forward({c, b})[0];
    EXPECT_EQ(c->data, 110);
    EXPECT_EQ(d->data, 210);
    d->Backward();
    EXPECT_EQ(d->grad, 1);
    EXPECT_EQ(c->grad, 1);
    EXPECT_EQ(b->grad, 2);
    EXPECT_EQ(a->grad, 1);
}

TEST(AutoDiffTest, RequiresGradFalseCase) {
    BasicSum<int, BasicBackend<int>> module;
    auto a = std::make_shared<Node<int>>(10, false);
    auto b = std::make_shared<Node<int>>(100, true);
    auto c = module.Forward({a, a})[0];
    auto d = module.Forward({a, b})[0];
    auto e = module.Forward({b, b})[0];
    auto f = module.Forward({c, d, e})[0];
    f->Backward();
    EXPECT_EQ(a->grad, 0);
    EXPECT_EQ(b->grad, 3);
    EXPECT_EQ(c->grad, 0);
    EXPECT_EQ(d->grad, 1);
    EXPECT_EQ(e->grad, 1);
}
