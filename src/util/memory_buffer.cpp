#include "memory_buffer.hpp"

#include <cstdlib>



namespace memory_buffer
{
    u8* malloc_bytes(size_t n_bytes)
    {
        auto data = std::malloc(n_bytes);
        assert(data);

        return (u8*)data;
    }


    void free_bytes(void* data)
    {
        std::free(data);
    }
}