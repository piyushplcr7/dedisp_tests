#include "dedisp_types.h"

void unpack(const dedisp_word* d_transposed,
            dedisp_size nsamps, dedisp_size nchan_words,
            dedisp_word* d_unpacked,
            dedisp_size in_nbits, dedisp_size out_nbits);
