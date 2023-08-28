#include <iostream>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <omp.h>
#include "packed.hpp"

#define PARENT(i) ((i - 1) / 2)
#define LCHILD(i) (2 * i + 1)
#define RCHILD(i) (2 * i + 2)

static int omp_thread_count() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}

class IndexableHeap {
    std::vector<packed_t> heap;
    std::vector<int> index_map;
    std::vector<omp_lock_t> locks;
    std::vector<unsigned> rand_states;
    packed_less less{};
    int num_par;

    inline bool left_shift(int pid) {
        return (rand_r(&rand_states[pid]) % 2) == 0;
    }

    void shift_down(int index, int pid) {
        if (left_shift(pid)) {
            int child = LCHILD(index);
            omp_set_lock(&locks[child]);
        } else {
            int child = RCHILD(index);
            omp_set_lock(&locks[child]);
        }
    }

  public:
    IndexableHeap(const std::vector<float> &init, int size)
    : heap(size), index_map(size), locks(size) {
        for (int i = 0; i < size; ++i) {
            heap[i] = pack(i, init[i]);
        }
        std::make_heap(heap.begin(), heap.end(), less);
        for (int i = 0; i < heap.size(); ++i) {
            auto packed = heap[i];
            int index = extract_int(packed);
            index_map[index] = i;
        }

        num_par = omp_thread_count();
        rand_states.resize(num_par);
        std::generate(rand_states.begin(), rand_states.end(), rand);
    }

    int num_threads() const { return num_par; }

    std::tuple<int, float> peek() const {
        return unpack(heap[0]);
    }

    // delta must be nonpositive
    void update(int index, float delta) {
        int pid = omp_get_thread_num();
        omp_set_lock(&locks[index]);
        // update current node
        int integer = extract_int(heap[index]);
        float score = extract_float(heap[index]);
        heap[index] = pack(integer, score + delta);
        // shift down
        while (true) {
            int lc = LCHILD(index);
            int rc = RCHILD(index);
            omp_set_lock(&locks[lc]);
            omp_set_lock(&locks[rc]);
            bool less_lc = less(heap[index], heap[lc]);
            bool less_rc = less(heap[index], heap[rc]);
            bool next_lc = left_shift(pid);
            if (less_lc && (next_lc || !less_rc)) {
                // choose left:
                // less_lc && less_rc && left_shift(pid) || (less_lc && !less_rc)
                continue;
            }
            if (less_rc && (!less_lc || !next_lc)) {
                // choose right:
                // less_lc && less_rc && !left_shift(pid) || (!less_lc && less_rc)
                continue;
            }
            // stay unchanged
            omp_unset_lock(&locks[lc]);
            omp_unset_lock(&locks[rc]);
            omp_unset_lock(&locks[index]);
            break;
        }
    }
};
