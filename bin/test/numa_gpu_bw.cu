#include <iostream>
#include <hwloc.h>
#include <cuda_runtime.h>
#include <numa.h>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " \
                  << cudaGetErrorString(err) << std::endl; \
        std::abort(); \
    } \
} while (0)

void touch_pages(void *ptr, size_t bytes)
{
    volatile char *p = (volatile char *)ptr;
    const size_t page = 4096;

    for (size_t i = 0; i < bytes; i += page) {
        p[i] = 0;
    }
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "usage: ./bw <mem_numa_node> <cuda_device>\n";
        return 1;
    }

    int mem_node = atoi(argv[1]);
    int cuda_dev = atoi(argv[2]);

    // ---------------- hwloc init ----------------
    hwloc_topology_t topo;
    hwloc_topology_init(&topo);
    hwloc_topology_load(topo);

    hwloc_obj_t req_numa =
        hwloc_get_obj_by_type(
            topo, HWLOC_OBJ_NUMANODE, mem_node);

    /* std::cout << "GPU-local NUMA: " << gpu_numa->logical_index
              << ", requested NUMA: " << mem_node << std::endl; */

    std::cout << "Pinning cpu and memory" << std::endl;
    // ---------------- CPU pinning ----------------
    hwloc_set_cpubind(
        topo,
        req_numa->cpuset,
        HWLOC_CPUBIND_THREAD
    );

    // ---------------- Memory binding ----------------
    hwloc_set_membind(
        topo,
        req_numa->nodeset,
        HWLOC_MEMBIND_BIND,
        HWLOC_MEMBIND_THREAD
    );

    // ---------------- CUDA device ----------------
    CUDA_CHECK(cudaSetDevice(cuda_dev));
    std::cout << "set cuda device " << cuda_dev << std::endl;

    /* cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, cuda_dev));

    std::cout << "properties of cude device:" << std::endl
        << "device_id = " << prop.pciDeviceID << std::endl
        << "domain_id = " << prop.pciDomainID << std::endl;

    // Find GPU PCI device in hwloc
    hwloc_obj_t gpu_pci = nullptr;
    for (hwloc_obj_t obj = hwloc_get_next_obj_by_type(
             topo, HWLOC_OBJ_PCI_DEVICE, nullptr);
         obj;
         obj = hwloc_get_next_obj_by_type(
             topo, HWLOC_OBJ_PCI_DEVICE, obj)) {

        if (obj->attr->pcidev.domain == prop.pciDomainID &&
            obj->attr->pcidev.bus    == prop.pciBusID &&
            obj->attr->pcidev.dev    == prop.pciDeviceID) {
            gpu_pci = obj;
            break;
        }
    }

    if (!gpu_pci) {
        std::cerr << "Could not find GPU in hwloc\n";
        return 1;
    }

    hwloc_obj_t cpu_ancestor =
        hwloc_get_non_io_ancestor_obj(topo, gpu_pci);

    hwloc_obj_t gpu_numa =
        hwloc_get_ancestor_obj_by_type(
            topo, HWLOC_OBJ_NUMANODE, cpu_ancestor); */

    

    // ---------------- Allocation ----------------
    size_t gigabytes = 15;
    const size_t bytes = gigabytes * 1024 * 1024 * 1024;

    std::cout << "Allocating pinned memory" << std::endl;
    
    /* void* hptr;
    CUDA_CHECK(cudaMallocHost(&hptr, bytes)); */

    void* hptr_h2d = numa_alloc_onnode(bytes, mem_node);
    void* hptr_d2h = numa_alloc_onnode(bytes, mem_node);

    std::cout << "Touching pages " << std::endl;
    touch_pages(hptr_h2d, bytes);
    touch_pages(hptr_d2h, bytes);

    CUDA_CHECK(cudaHostRegister(hptr_h2d, bytes, cudaHostRegisterPortable));
    CUDA_CHECK(cudaHostRegister(hptr_d2h, bytes, cudaHostRegisterPortable));

    void* dptr_h2d;
    void* dptr_d2h;
    CUDA_CHECK(cudaMalloc(&dptr_h2d, bytes));
    CUDA_CHECK(cudaMalloc(&dptr_d2h, bytes));

    std::cout << "Verifying NUMA placement" << std::endl;
    // Verify NUMA placement
    hwloc_nodeset_t ns = hwloc_bitmap_alloc();
    hwloc_membind_policy_t policy;

    int err = hwloc_get_area_membind(topo, hptr_h2d, bytes, ns, &policy, HWLOC_MEMBIND_BYNODESET);
    if (err) {
        std::cerr << "hwloc_get_area_membind failed: " << strerror(errno) << std::endl;
    } else {
        char buf[128];
        hwloc_bitmap_snprintf(buf, sizeof(buf), ns);
        std::cout << "Host buffer NUMA nodeset: " << buf << std::endl;
    }

    err = hwloc_get_area_membind(topo, hptr_d2h, bytes, ns, &policy, HWLOC_MEMBIND_BYNODESET);
    if (err) {
        std::cerr << "hwloc_get_area_membind failed: " << strerror(errno) << std::endl;
    } else {
        char buf[128];
        hwloc_bitmap_snprintf(buf, sizeof(buf), ns);
        std::cout << "Host buffer NUMA nodeset: " << buf << std::endl;
    }
    hwloc_bitmap_free(ns);

    // ---------------- Timing ----------------
    int trials = 10;

    std::vector<cudaEvent_t> starth2d(trials), stoph2d(trials);
    std::vector<cudaEvent_t> startd2h(trials), stopd2h(trials);

    for (int i = 0 ; i < trials ; ++i) {
        cudaEventCreate(&starth2d[i]);
        cudaEventCreate(&stoph2d[i]);
        
        cudaEventCreate(&startd2h[i]);
        cudaEventCreate(&stopd2h[i]);
    }
    

    // ---------------- Streams ----------------
    cudaStream_t streamH2D, streamD2H;
    CUDA_CHECK(cudaStreamCreate(&streamH2D));
    //streamD2H = streamH2D;
    CUDA_CHECK(cudaStreamCreate(&streamD2H));

    // Warmup
    CUDA_CHECK(cudaMemset(dptr_h2d, 0, bytes));
    CUDA_CHECK(cudaMemset(dptr_d2h, 0, bytes));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpyAsync(dptr_h2d, hptr_h2d, bytes, cudaMemcpyHostToDevice, streamH2D));
    CUDA_CHECK(cudaMemcpyAsync(hptr_d2h, dptr_d2h, bytes, cudaMemcpyDeviceToHost, streamD2H));
    CUDA_CHECK(cudaDeviceSynchronize());


    for (int i = 0 ; i < trials ; ++i) {
        CUDA_CHECK(cudaEventRecord(starth2d[i], streamH2D));
        CUDA_CHECK(cudaMemcpyAsync(dptr_h2d, hptr_h2d, bytes, cudaMemcpyHostToDevice, streamH2D));
        CUDA_CHECK(cudaEventRecord(stoph2d[i], streamH2D));

        CUDA_CHECK(cudaEventRecord(startd2h[i], streamD2H));
        CUDA_CHECK(cudaMemcpyAsync(hptr_d2h, dptr_d2h, bytes, cudaMemcpyDeviceToHost, streamD2H));
        CUDA_CHECK(cudaEventRecord(stopd2h[i], streamD2H));
    }
    
    CUDA_CHECK(cudaStreamSynchronize(streamH2D));
    CUDA_CHECK(cudaStreamSynchronize(streamD2H));
    
    float totmsh2d=0., totmsd2h=0.;
    float ms;
    for (int i = 0 ; i < trials ; ++i) {
        CUDA_CHECK(cudaEventElapsedTime(&ms, starth2d[i], stoph2d[i]));
        totmsh2d += ms;

        CUDA_CHECK(cudaEventElapsedTime(&ms, startd2h[i], stopd2h[i]));
        totmsd2h += ms;
    }
    
    totmsh2d /= trials;
    totmsd2h /= trials;

    std::cout << "H2D: " << gigabytes / (totmsh2d/1e3) << " GBps" << std::endl;
    std::cout << "D2H: " << gigabytes / (totmsd2h/1e3) << " GBps" << std::endl;

    // ---------------- Cleanup ----------------
    for (int i = 0 ; i < trials ; ++i) {
        CUDA_CHECK(cudaEventDestroy(starth2d[i]));
        CUDA_CHECK(cudaEventDestroy(stoph2d[i]));
        CUDA_CHECK(cudaEventDestroy(startd2h[i]));
        CUDA_CHECK(cudaEventDestroy(stopd2h[i]));
    }
    CUDA_CHECK(cudaStreamDestroy(streamH2D));
    CUDA_CHECK(cudaStreamDestroy(streamD2H));

    cudaFree(dptr_h2d);
    cudaFree(dptr_d2h);
    //cudaFreeHost(hptr);
    CUDA_CHECK(cudaHostUnregister(hptr_h2d));
    numa_free(hptr_h2d, bytes);

    CUDA_CHECK(cudaHostUnregister(hptr_d2h));
    numa_free(hptr_d2h, bytes);
    
    return 0;
}
