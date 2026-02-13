import os

file_path = 'c:/Users/sao/Documents/rra/cpp_loader_optimized.cpp'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Define dt in float kernel
content = content.replace(
    'const int64_t t_bins = runtime_bins;\n\n    Perf::g_lgh_calls',
    'const int64_t t_bins = runtime_bins;\n    const int64_t dt = base_phase;\n\n    Perf::g_lgh_calls'
)
# Fallback if double newline is different
content = content.replace(
    'const int64_t t_bins = runtime_bins;\r\n\r\n    Perf::g_lgh_calls',
    'const int64_t t_bins = runtime_bins;\r\n    const int64_t dt = base_phase;\r\n\r\n    Perf::g_lgh_calls'
)

# 2. Define dt in int8 kernel
content = content.replace(
    'const int64_t t_bins = runtime_bins;\n\n    Perf::g_lgh_calls',
    'const int64_t t_bins = runtime_bins;\n    const int64_t dt = base_phase;\n\n    Perf::g_lgh_calls'
)

# 3. Remove residual morton_inverse_order checks
content = content.replace('    check_cpu(morton_inverse_order, "morton_inverse_order");\n', '')
content = content.replace('    check_cpu(morton_inverse_order, "morton_inverse_order");\r\n', '')
content = content.replace('    check_dim(morton_inverse_order, 1, "morton_inverse_order");\n', '')
content = content.replace('    check_dim(morton_inverse_order, 1, "morton_inverse_order");\r\n', '')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(">>> C++ fix script completed.")
