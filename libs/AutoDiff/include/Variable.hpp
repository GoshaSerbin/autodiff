#pragma once

#include <functional>
#include <memory>
#include <vector>

namespace ai {
    template <typename T>
    class Variable {
     public:
        T value;
        T grad;
        bool requiresGrad;
        std::vector<std::shared_ptr<Variable>> parents;
        std::function<void()> backwardFn;

        Variable(T value, bool requiresGrad = false) : value(value), grad(value), requiresGrad(requiresGrad) { grad = 0; }

        auto Backward() -> void;
    };

    template <typename T>
    std::shared_ptr<Variable<T>> operator+(const std::shared_ptr<Variable<T>>& a, const std::shared_ptr<Variable<T>>& b) {
        auto out = std::make_shared<Variable<T>>(a->value + b->value, a->requiresGrad || b->requiresGrad);
        if (out->requiresGrad) {
            out->parents = {a, b};
            out->backwardFn = [a = a.get(), b = b.get(), out = out.get()]() {
                if (a->requiresGrad) a->grad += out->grad;
                if (b->requiresGrad) b->grad += out->grad;
            };
        }
        return out;
    }

    template <typename T>
    std::shared_ptr<Variable<T>> operator-(const std::shared_ptr<Variable<T>>& a, const std::shared_ptr<Variable<T>>& b) {
        auto out = std::make_shared<Variable<T>>(a->value - b->value, a->requiresGrad || b->requiresGrad);
        if (out->requiresGrad) {
            out->parents = {a, b};
            out->backwardFn = [a = a.get(), b = b.get(), out = out.get()]() {
                if (a->requiresGrad) a->grad += out->grad;
                if (b->requiresGrad) b->grad -= out->grad;
            };
        }
        return out;
    }

    template <typename T>
    std::shared_ptr<Variable<T>> operator*(const std::shared_ptr<Variable<T>>& a, const std::shared_ptr<Variable<T>>& b) {
        auto out = std::make_shared<Variable<T>>(a->value * b->value, a->requiresGrad || b->requiresGrad);
        if (out->requiresGrad) {
            out->parents = {a, b};
            out->backwardFn = [a = a.get(), b = b.get(), out = out.get()]() {
                if (a->requiresGrad) a->grad += b->value * out->grad;
                if (b->requiresGrad) b->grad += a->value * out->grad;
            };
        }
        return out;
    }

    template <typename T>
    std::shared_ptr<Variable<T>> operator/(const std::shared_ptr<Variable<T>>& a, const std::shared_ptr<Variable<T>>& b) {
        auto out = std::make_shared<Variable<T>>(a->value / b->value, a->requiresGrad || b->requiresGrad);
        if (out->requiresGrad) {
            out->parents = {a, b};
            out->backwardFn = [a = a.get(), b = b.get(), out = out.get()]() {
                if (a->requiresGrad) a->grad += (1.0 / b->value) * out->grad;
                if (b->requiresGrad) b->grad -= (a->value / (b->value * b->value)) * out->grad;
            };
        }
        return out;
    }

    template <typename T>
    std::shared_ptr<Variable<T>> operator-(const std::shared_ptr<Variable<T>>& a) {
        auto out = std::make_shared<Variable<T>>(-(a->value), a->requiresGrad);
        if (out->requiresGrad) {
            out->parents = {a};
            out->backwardFn = [a = a.get(), out = out.get()]() {
                if (a->requiresGrad) a->grad -= out->grad;
            };
        }
        return out;
    }

    template <typename T>
    void Variable<T>::Backward() {
        if (!requiresGrad) return;

        grad = 1;
        std::vector<Variable*> topo_order;
        std::unordered_set<Variable*> visited;

        std::function<void(Variable*)> build_topo = [&](Variable* node) {
            if (visited.count(node) || !node->requiresGrad) return;
            visited.insert(node);
            for (auto& parent : node->parents) build_topo(parent.get());
            topo_order.push_back(node);
        };

        build_topo(this);

        for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
            if ((*it)->backwardFn) (*it)->backwardFn();
        }
    }

}  // namespace ai
