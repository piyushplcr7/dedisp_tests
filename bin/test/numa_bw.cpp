#include <numa.h>
#include <numaif.h>
#include <sched.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <chrono>

static void pin_cpu_node(int node) {
    cpu_set_t set;
    CPU_ZERO(&set);

    struct bitmask* cpus = numa_allocate_cpumask();
    numa_node_to_cpus(node, cpus);

    for (int i = 0; i < cpus->size; i++) {
        if (numa_bitmask_isbitset(cpus, i)) {
            CPU_SET(i, &set);
        }
    }

    sched_setaffinity(0, sizeof(set), &set);
    numa_free_cpumask(cpus);
}

static void pin_cpu_node_first(int node) {
    cpu_set_t set;
    CPU_ZERO(&set);

    struct bitmask* cpus = numa_allocate_cpumask();
    numa_node_to_cpus(node, cpus);

    for (int i = 0; i < cpus->size; i++) {
        if (numa_bitmask_isbitset(cpus, i)) {
            CPU_SET(i, &set);
            break;
        }
    }

    sched_setaffinity(0, sizeof(set), &set);
    numa_free_cpumask(cpus);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " <cpu_node> <mem_node>\n";
        return 1;
    }

    int cpu_node = atoi(argv[1]);
    int mem_node = atoi(argv[2]);

    const size_t bytes = 20UL * 1024 * 1024 * 1024; // 4 GB
    const size_t n = bytes / sizeof(uint64_t);

    pin_cpu_node_first(cpu_node);

    uint64_t* buf = (uint64_t*)numa_alloc_onnode(bytes, mem_node);
    if (!buf) {
        perror("numa_alloc_onnode");
        return 1;
    }

    // First touch: ensure pages are allocated
    for (size_t i = 0; i < n; i += 64) {
        buf[i] = 1;
    }

    volatile uint64_t sum = 0;

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n; i += 8) {
        sum += buf[i];
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> dt = end - start;

    double bw = (double)bytes / dt.count() / 1e9;

    std::cout << "CPU node " << cpu_node
              << ", MEM node " << mem_node
              << " → " << bw << " GB/s, sum=" << sum << "\n";

    numa_free(buf, bytes);
    return 0;
}
