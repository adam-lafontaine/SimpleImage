#include "execute.hpp"

#include <cassert>

#ifndef SIMAGE_NO_PARALLEL
/*
constexpr u32 N_THREADS = 16;

template <class LIST_T, class FUNC_T>
static void do_for_each(LIST_T const& list, FUNC_T const& func)
{
	std::for_each(std::execution::par, list.begin(), list.end(), func);
}

class ThreadProcess
{
public:
	u32 thread_id = 0;
	id_func_t process;
};


using ProcList = std::array<ThreadProcess, N_THREADS>;


static ProcList make_proc_list(id_func_t const& id_func)
{
	ProcList list = { 0 };

	for (u32 i = 0; i < N_THREADS; ++i)
	{
		list[i] = { i, id_func };
	}

	return list;
}


static void execute_procs(ProcList const& list)
{
	auto const func = [](ThreadProcess const& t) { t.process(t.thread_id); };

	do_for_each(list, func);
}


void process_range_old(u32 id_begin, u32 id_end, id_func_t const& id_func)
{
    assert(id_begin <= id_end);

    auto const n_per_thread = (id_end - id_begin) / N_THREADS;

    auto const thread_func = [&](u32 t)
    {
        auto const n_begin = t * n_per_thread + id_begin;
        auto const n_end = (t == N_THREADS - 1) ? id_end : (t + 1) * n_per_thread;

        for (u32 id = n_begin; id < n_end; ++id)
        {
            id_func(id);
        }
    };

    execute_procs(make_proc_list(thread_func));
}
*/

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