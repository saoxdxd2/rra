import os
import re

file_path = 'c:/Users/sao/Documents/rra/cpp_loader_optimized.cpp'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. & 2. Define dt in both kernels using regex
# Find 'const int64_t t_bins = runtime_bins;' followed by whitespace and 'Perf::g_lgh_calls'
pattern = re.compile(r'(const int64_t t_bins = runtime_bins;\s+)(Perf::g_lgh_calls)')
content = pattern.sub(r'\1const int64_t dt = base_phase;\n\n    \2', content)

# 3. Remove residual morton_inverse_order checks
content = re.sub(r'^\s*check_cpu\(morton_inverse_order, "morton_inverse_order"\);\r?\n', '', content, flags=re.MULTILINE)
content = re.sub(r'^\s*check_dim\(morton_inverse_order, 1, "morton_inverse_order"\);\r?\n', '', content, flags=re.MULTILINE)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(">>> C++ fix script (regex) completed.")
