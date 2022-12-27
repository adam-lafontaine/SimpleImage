#pragma once

#include "../defines.hpp"

#include <algorithm>
#include <functional>
#include <array>
#include <vector>


#ifndef SIMPLE_NO_PARALLEL

#include <execution>
// -ltbb


template <size_t N>
inline void execute_parallel(std::array<std::function<void()>, N> const& f_list)
{
    std::for_each(std::execution::par, f_list.begin(), f_list.end(), [](auto const& f){ f(); });
}


inline void execute_parallel(std::vector<std::function<void()>> const& f_list)
{
    std::for_each(std::execution::par, f_list.begin(), f_list.end(), [](auto const& f){ f(); });
}

#endif // !SIMPLE_NO_PARALLEL


template <size_t N>
inline void execute_sequential(std::array<std::function<void()>, N> const& f_list)
{
    std::for_each(f_list.begin(), f_list.end(), [](auto const& f){ f(); });
}


inline void execute_sequential(std::vector<std::function<void()>> const& f_list)
{
    std::for_each(f_list.begin(), f_list.end(), [](auto const& f){ f(); });
}


template <size_t N>
inline void execute(std::array<std::function<void()>, N> const& f_list)
{
    execute_parallel(f_list);
    //execute_sequential(f_list);
}


inline void execute(std::vector<std::function<void()>> const& f_list)
{
    execute_parallel(f_list);
    //execute_sequential(f_list);
}


using id_func_t = std::function<void(u32)>;


void process_range(u32 id_begin, u32 id_end, id_func_t const& id_func);