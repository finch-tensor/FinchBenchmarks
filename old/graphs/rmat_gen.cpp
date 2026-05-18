#include <iostream>
#include <fstream>
#include <set>
#include <vector>
#include <random>
#include <algorithm>

void generate_rmat_graph(int scale, int avg_edges_per_vertex) {
    int num_vertices = 1 << scale;  // 2^scale
    int num_edges = num_vertices * avg_edges_per_vertex;

    // Using a set to store unique edges (src, dst)
    std::set<std::pair<int, int>> edges;

    // Random number generator for weights
    std::default_random_engine generator;
    std::uniform_real_distribution<double> weight_distribution(0.0, 1.0); // Weights between 0 and 1

    // Edge generation parameters
    double a = 0.57, b = 0.19, c = 0.19, d = 0.05;

    while (edges.size() < num_edges) {
        int src = 1, dst = 1;
        int stride = num_vertices / 2;

        for (int i = 0; i < scale; ++i) {
            double rand_val = static_cast<double>(rand()) / RAND_MAX;
            if (rand_val < a) {
                // Stay in top-left quadrant
            } else if (rand_val < a + b) {
                // Move to top-right quadrant
                dst += stride;
            } else if (rand_val < a + b + c) {
                // Move to bottom-left quadrant
                src += stride;
            } else {
                // Move to bottom-right quadrant
                src += stride;
                dst += stride;
            }
            stride /= 2;
        }

        // Ensure src and dst are different
        if (src != dst) {
            edges.emplace(src, dst);
        }
    }

    // Sort edges based on src, then dst
    std::vector<std::pair<int, int>> sorted_edges(edges.begin(), edges.end());

    // Write to .mtx file
    std::ofstream outfile("rmat_s" + std::to_string(scale) + "_e" + std::to_string(avg_edges_per_vertex) + ".mtx");
    if (outfile.is_open()) {
        outfile << "%%MatrixMarket matrix coordinate real general\n";
        outfile << num_vertices << " " << num_vertices << " " << sorted_edges.size() << "\n";

        for (const auto& edge : sorted_edges) {
            int src = edge.first;
            int dst = edge.second;
            double weight = weight_distribution(generator);  // Generate weight for the edge (0 to 1)
            outfile << src << " " << dst << " " << weight << "\n"; // Write src, dst, and weight
        }
        outfile.close();
    } else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }
}

int main(int argc, char* args[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << args[0] << " <scale> <avg_edges_per_vertex>" << std::endl;
        return 1;
    }

    int scale = std::stoi(args[1]);
    int avg_edges_per_vertex = std::stoi(args[2]);

    generate_rmat_graph(scale, avg_edges_per_vertex);
    return 0;
}
