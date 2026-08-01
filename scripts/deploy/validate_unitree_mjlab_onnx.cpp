#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "isaaclab/algorithms/algorithms.h"

namespace
{

std::vector<std::vector<float>> read_rows(const std::string& path, const size_t expected_width)
{
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("Could not open input file: " + path);
    }

    std::vector<std::vector<float>> rows;
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream line_stream(line);
        std::vector<float> row;
        float value = 0.0f;
        while (line_stream >> value) {
            row.push_back(value);
        }
        if (row.size() != expected_width) {
            throw std::runtime_error(
                path + " row has " + std::to_string(row.size()) +
                " values, expected " + std::to_string(expected_width) + "."
            );
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 2 && argc != 6) {
        std::cerr
            << "Usage: validate_unitree_mjlab_onnx <policy.onnx>\n"
            << "   or: validate_unitree_mjlab_onnx <policy.onnx> "
               "<policy_obs.txt> <policy_history.txt> <expected_action.txt> <tolerance>\n";
        return 2;
    }

    try {
        isaaclab::OrtRunner runner(argv[1]);
        if (argc == 6) {
            const auto policy_obs = read_rows(argv[2], 45);
            const auto policy_history = read_rows(argv[3], 4500);
            const auto expected_action = read_rows(argv[4], 12);
            const float tolerance = std::stof(argv[5]);
            if (policy_obs.size() != policy_history.size() ||
                policy_obs.size() != expected_action.size()) {
                throw std::runtime_error("Golden input files have different row counts.");
            }

            float max_abs_error = 0.0f;
            for (size_t case_idx = 0; case_idx < policy_obs.size(); ++case_idx) {
                std::unordered_map<std::string, std::vector<float>> obs{
                    {"policy_obs", policy_obs[case_idx]},
                    {"policy_history", policy_history[case_idx]},
                };
                const auto action = runner.act(std::move(obs));
                for (size_t action_idx = 0; action_idx < action.size(); ++action_idx) {
                    max_abs_error = std::max(
                        max_abs_error,
                        std::abs(action[action_idx] - expected_action[case_idx][action_idx])
                    );
                }
            }
            std::cout << "Golden ONNX cases: " << policy_obs.size() << '\n';
            std::cout << "Maximum absolute error: " << max_abs_error << '\n';
            std::cout << "Tolerance: " << tolerance << '\n';
            if (max_abs_error > tolerance) {
                std::cerr << "ONNX golden inference parity failed.\n";
                return 1;
            }
            std::cout << "ONNX golden inference parity passed.\n";
            return 0;
        }

        std::unordered_map<std::string, std::vector<float>> obs{
            {"policy_obs", std::vector<float>(45, 0.0f)},
            {"policy_history", std::vector<float>(4500, 0.0f)},
        };
        const auto action = runner.act(std::move(obs));
        if (action.size() != 12) {
            std::cerr << "Expected 12 actions, got " << action.size() << "\n";
            return 1;
        }
        for (const float value : action) {
            if (!std::isfinite(value)) {
                std::cerr << "Policy produced a non-finite action.\n";
                return 1;
            }
        }

        std::cout << "ONNX contract OK: policy_obs=45 policy_history=4500 action=12\n";
        std::cout << "Zero-input action:";
        for (const float value : action) {
            std::cout << ' ' << value;
        }
        std::cout << '\n';
    } catch (const std::exception& exc) {
        std::cerr << "ONNX validation failed: " << exc.what() << '\n';
        return 1;
    }
    return 0;
}
