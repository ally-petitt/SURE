#include <iostream>
#include <queue>
#include <algorithm>
#include <vector>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <iomanip>

// Structure to store item information
struct Item {
    float weight; // Weight of the item
    int value;    // Value of the item
};

// Structure to represent a node in the decision tree
struct Node {
    int level;              // Level of the node (index in the item array)
    int profit;             // Profit accumulated so far
    int bound;              // Upper bound of maximum profit in subtree
    float weight;           // Weight accumulated so far
    bool selectedItems[100]; // Items selected up to this point
};

// Comparison function to sort items by value-to-weight ratio
__host__ __device__
bool cmp(const Item &a, const Item &b) {
    double r1 = (double)a.value / a.weight;
    double r2 = (double)b.value / b.weight;
    return r1 > r2;
}

// Calculate the upper bound on maximum profit (bound) for a node
__device__
int d_bound(Node u, int n, int W, Item *d_arr) {
    // If weight exceeds capacity, bound is 0
    if (u.weight >= W)
        return 0;

    int profit_bound = u.profit; // Start with the current profit
    int j = u.level + 1;         // Next item index
    float totweight = u.weight;  // Start with the current weight

    // Add items while the total weight is less than capacity
    while ((j < n) && (totweight + d_arr[j].weight <= W)) {
        totweight += d_arr[j].weight;
        profit_bound += d_arr[j].value;
        j++;
    }

    // If there is still space, add a fraction of the next item's value
    if (j < n)
        profit_bound += (W - totweight) * d_arr[j].value / d_arr[j].weight;

    return profit_bound;
}

// CUDA kernel to calculate bounds for nodes
__global__
void calculate_bounds(Node *nodes, int n, int W, Item *d_arr, int *d_maxProfit, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        Node u = nodes[idx];
        // Calculate the bound of the current node
        nodes[idx].bound = d_bound(u, n, W, d_arr);
        // Update the maximum profit found so far
        if (nodes[idx].bound > *d_maxProfit) {
            atomicMax(d_maxProfit, nodes[idx].profit);
        }
    }
}

// Function to solve the knapsack problem using the Branch and Bound method
void knapsack(int W, Item *h_arr, int n) {
    // Sort items based on value-to-weight ratio in descending order
    thrust::sort(h_arr, h_arr + n, cmp);

    // Allocate memory on the device for items
    Item *d_arr;
    cudaMalloc(&d_arr, n * sizeof(Item));
    cudaMemcpy(d_arr, h_arr, n * sizeof(Item), cudaMemcpyHostToDevice);

    // Initialize maximum profit
    int h_maxProfit = 0;
    int *d_maxProfit;
    cudaMalloc(&d_maxProfit, sizeof(int));
    cudaMemcpy(d_maxProfit, &h_maxProfit, sizeof(int), cudaMemcpyHostToDevice);

    // Initialize root node of the decision tree
    Node u, v;
    u.level = -1;
    u.profit = u.weight = 0;
    memset(u.selectedItems, false, sizeof(u.selectedItems));

    // Queue to store nodes for BFS traversal of the decision tree
    std::queue<Node> Q;
    Q.push(u);

    // To store the best selection of items
    std::vector<bool> bestSelectedItems(100, false);
    int totalNodes = 0;  // Counter for total nodes processed
    int validNodes = 0;  // Counter for nodes that were not pruned

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // BFS traversal of the decision tree
    while (!Q.empty()) {
        u = Q.front();
        Q.pop();
        totalNodes++;

        // If this is the starting node, set level to 0
        if (u.level == -1)
            v.level = 0;

        // If we have reached the last item, skip further exploration
        if (u.level == n - 1)
            continue;

        v.level = u.level + 1;

        // Include the current item
        v.weight = u.weight + h_arr[v.level].weight;
        v.profit = u.profit + h_arr[v.level].value;
        memcpy(v.selectedItems, u.selectedItems, sizeof(u.selectedItems));
        v.selectedItems[v.level] = true;

        // Update max profit and best selection if we found a better solution
        if (v.weight <= W && v.profit > h_maxProfit) {
            h_maxProfit = v.profit;
            bestSelectedItems.assign(v.selectedItems, v.selectedItems + 100);
        }

        // Explore the node including and excluding the current item
        Node nodes[2];
        nodes[0] = v;

        // Exclude the current item
        nodes[1].level = v.level;
        nodes[1].weight = u.weight;
        nodes[1].profit = u.profit;
        memcpy(nodes[1].selectedItems, u.selectedItems, sizeof(u.selectedItems));

        // Allocate memory for nodes on the device
        Node *d_nodes;
        cudaMalloc(&d_nodes, 2 * sizeof(Node));
        cudaMemcpy(d_nodes, nodes, 2 * sizeof(Node), cudaMemcpyHostToDevice);

        int blockSize = 2;
        int numBlocks = (2 + blockSize - 1) / blockSize;
        calculate_bounds<<<numBlocks, blockSize>>>(d_nodes, n, W, d_arr, d_maxProfit, 2);

        cudaMemcpy(&h_maxProfit, d_maxProfit, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(nodes, d_nodes, 2 * sizeof(Node), cudaMemcpyDeviceToHost);

        // Only explore nodes with a bound greater than current max profit
        if (nodes[0].bound > h_maxProfit) {
            Q.push(nodes[0]);
            validNodes++;
        }
        if (nodes[1].bound > h_maxProfit) {
            Q.push(nodes[1]);
            validNodes++;
        }

        cudaFree(d_nodes);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(d_arr);
    cudaFree(d_maxProfit);

    int prunedNodes = totalNodes - validNodes;

    // Output results
    std::cout << "Maximum possible profit = " << h_maxProfit << std::endl;
    std::cout << "Items selected (0-indexed):" << std::endl;
    float totalWeight = 0.0f;
    for (int i = 0; i < n; i++) {
        if (bestSelectedItems[i]) {
            std::cout << i << " (Value: " << h_arr[i].value << ", Weight: " << h_arr[i].weight << ")" << std::endl;
            totalWeight += h_arr[i].weight;
        }
    }
    std::cout << "Total Weight: " << totalWeight << std::endl;
    std::cout << "Time taken: " << milliseconds << " ms" << std::endl;
    std::cout << "Total Nodes: " << totalNodes << std::endl;
    std::cout << "Valid Nodes: " << validNodes << std::endl;
    std::cout << "Pruned Nodes: " << prunedNodes << std::endl;
}

int main() {
    int W = 50; // Capacity of the knapsack
    Item arr[] = {
        {2, 40}, {3.14, 50}, {1.98, 100}, {5, 95}, {3, 30},
        {4, 70}, {6, 60}, {3.5, 75}, {2.5, 85}, {4.2, 60},
        {1.3, 20}, {5.5, 80}, {7, 120}, {1.5, 10}, {6.5, 90},
        {2.2, 40}, {3.8, 50}, {1.9, 100}, {5.4, 95}, {3.1, 30},
        {4.6, 70}, {6.2, 60}, {3.9, 75}, {2.7, 85}, {4.3, 60},
        {1.6, 20}, {5.7, 80}, {7.1, 120}, {1.8, 10}, {6.8, 90}
    };
    int n = sizeof(arr) / sizeof(arr[0]);

    knapsack(W, arr, n);

    return 0;
}
