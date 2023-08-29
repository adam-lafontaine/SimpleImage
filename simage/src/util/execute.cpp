#include "execute.hpp"

#include <cassert>

#ifndef SIMAGE_NO_PARALLEL

void process_range(u32 id_begin, u32 id_end, id_func_t const& id_func)
{
    auto [min, max] = std::minmax(id_begin, id_end);

    std::vector<u32> list(max - min);
    std::iota(list.begin(), list.end(), min);

    std::for_each(std::execution::par, list.begin(), list.end(), id_func);
}


#else

void process_range(u32 id_begin, u32 id_end, id_func_t const& id_func)
{
    assert(id_begin <= id_end);

    for (u32 i = id_begin; i < id_end; ++i)
    {
        id_func(i);
    }
}

#endif // !SIMAGE_NO_PARALLEL