/**
 * @file Node.hpp
 * @brief Defines the Node class for automatic differentiation.
 *
 * This file contains the implementation of the Node class, which represents
 * a node in a computational graph for automatic differentiation. Each node
 * holds data, gradient information, and dependencies on parent nodes.
 */

#pragma once

#include <concepts>
#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

namespace auto_diff {

    /**
     * @brief Concept for Node inner data type
     *
     * This concept enforces assignability and assign-from-int.
     * This concept requires that a type T supports assignment from another T and from an integer literal.
     * This requirements are needed to enable using Node with primitive data types, e.g. `int` or `float`.
     * In Node constructor `grad` is assigned to `data` (to reshape it) and then assigned to `0`. In `Backward` pass `grad` is assigned to
     * `0`
     */
    template <typename T>
    concept NodeT = requires(T a, T b) {
        { a = b } -> std::same_as<T&>;
        { a = 1 } -> std::same_as<T&>;
    };

    template <NodeT T>
    class Node;

    /**
     * @brief A shared ownership pointer to a Node
     *
     * Smart pointer is used to garantee that the node is alive if needed and can be deleted if not. Shared ownership is chosen because the
     * same node can be required by many others. For example \f$c = a * b\f$ and \f$d = c * a\f$ means that \f$a\f$ is required for both
     * \f$c\f$ and \f$d\f$ nodes during the backpropagation.
     */
    template <NodeT T>
    using NodePtr = std::shared_ptr<Node<T>>;

    template <NodeT T>
    using NodePtrVector = std::vector<NodePtr<T>>;

    /**
     * @class Node
     * @brief Represents a node in a computational graph for automatic differentiation.
     *
     * @tparam T The data type of the node's value and gradient. Must satisfy NodeT concept
     *
     * The Node class is used to represent a single computation in a computational
     * graph. It stores the value of the computation, its gradient, and references
     * to its parent nodes. The class also provides functionality for backpropagation
     * and topological sorting of the graph.
     */
    template <NodeT T>
    class Node {
     public:
        Node() = default;
        Node(T data, bool requiresGrad = true) : data(data), requiresGrad(requiresGrad) {
            if (requiresGrad) {
                grad = data;
                grad = 0;
            }
        }

        /**
         * @brief Performs backpropagation starting from this node.
         *
         * This function computes the gradients for all nodes in the computational
         * graph by traversing the graph in reverse topological order. It starts
         * by setting the gradient of the current node to 1 and then propagates
         * gradients backward through the graph.
         */
        auto Backward() -> void {
            if (!requiresGrad) return;
            grad = 1;
            std::vector<Node<T>*> topologicalSortedNodes = TopologicalSort();

            for (auto it = topologicalSortedNodes.rbegin(); it != topologicalSortedNodes.rend(); ++it) {
                if ((*it)->backwardFn) {
                    (*it)->backwardFn();
                }
            }
        }

        T data{};
        T grad{};
        bool requiresGrad{true};
        std::function<void()> backwardFn;
        std::vector<NodePtr<T>> parents;

     private:
        /**
         * @brief Performs a topological sort of the computational graph.
         *
         * @return A vector of pointers to nodes in topological order.
         *
         * This function sorts the nodes in the computational graph in topological
         * order, ensuring that parent nodes appear before their children.
         */
        auto TopologicalSort() -> std::vector<Node<T>*> {
            std::vector<Node<T>*> sortedNodes;
            std::unordered_set<Node<T>*> visited;
            TopologicalSortInner(this, visited, sortedNodes);
            return sortedNodes;
        }

        /**
         * @brief Helper function for topological sorting.
         *
         * @param node The current node being visited.
         * @param visited A set of nodes that have already been visited.
         * @param sortedNodes A vector to store the sorted nodes.
         *
         * This function recursively visits all parent nodes of the given node
         * and adds them to the sorted list in topological order.
         */
        auto TopologicalSortInner(Node<T>* node, std::unordered_set<Node<T>*>& visited, std::vector<Node<T>*>& sortedNodes) -> void {
            if (visited.count(node) || !node->requiresGrad) return;
            visited.insert(node);
            for (auto& parent : node->parents) {
                TopologicalSortInner(parent.get(), visited, sortedNodes);
            }
            sortedNodes.push_back(node);
        }
    };

}  // namespace auto_diff
