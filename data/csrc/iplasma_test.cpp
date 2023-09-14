#include <cstdlib>
#include <iostream>
#include "mimalloc.h"

int main(void) {
    std::cout << "mimalloc version: " << mi_version() << "\n";
    constexpr size_t memsize = 1024 * 1024 * 1024;
    void *mem = 0;
    posix_memalign(&mem, memsize, memsize);
    std::cout << std::hex << "[" <<  mem << ", " << (void *)((char *)mem+memsize) << ")\n";

    mi_arena_id_t mi_arena;
    bool success = mi_manage_os_memory_ex(mem, memsize, true, true, false, -1, true, &mi_arena);
    std::cout << "arena: " << mi_arena << "\n";
    mi_manage_os_memory_ex(mem, memsize, true, true, false, -1, true, &mi_arena);
    std::cout << "arena: " << mi_arena << "\n";
    mi_heap_t *heap = mi_heap_new_in_arena(mi_arena);
    if (heap == nullptr) {
        std::cerr << "heap is null!\n";
    }
    mi_option_set(mi_option_limit_os_alloc, 1);
    if (!success) {
        std::cout << "mi_manage_os_memory failed\n";
    } else {
        // void *p1 = mi_heap_malloc(heap, 1024 * 1024);
        // std::cout << std::hex << p1 << "\n";
        // void *p2 = mi_heap_malloc(heap, 1024 * 1024 * 512);
        // std::cout << std::hex << p2 << "\n";
        // void *p3 = mi_heap_malloc(heap, 1024 * 1024 * 128);
        // std::cout << std::hex << p3 << "\n";
        // void *p4 = mi_heap_malloc(heap, 1024 * 1024 * 128);
        // std::cout << std::hex << p4 << "\n";
        void *p1 = mi_malloc(1024 * 1024);
        std::cout << std::hex << p1 << "\n";
        void *p2 = mi_malloc(1024 * 1024 * 512);
        std::cout << std::hex << p2 << "\n";
        void *p3 = mi_malloc(1024 * 1024 * 128);
        std::cout << std::hex << p3 << "\n";
        void *p4 = mi_malloc(1024 * 1024 * 128);
        std::cout << std::hex << p4 << "\n";
    }
    free(mem);
    return 0;
}
