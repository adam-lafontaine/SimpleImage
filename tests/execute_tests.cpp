#include "../src/util/execute.hpp"
#include "../src/util/stopwatch.hpp"

#include <cstdio>
#include <thread>
#include <mutex>
#include <cstdarg>


std::mutex console_mtx;

 static void console_print(const char* format, ...)
 {
    std::lock_guard<std::mutex> lock(console_mtx);

    va_list args;
    va_start(args, format);

    vprintf(format, args);            

    va_end(args);
 }


static void sleep_id(int id, int sleep_ms)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    console_print("%d ", id);
}


static std::array<std::function<void()>, 10> make_func_array()
{
    constexpr int sleep_ms = 5;

    std::array<std::function<void()>, 10> f_array
    {
        [&](){ sleep_id(1, sleep_ms); },
        [&](){ sleep_id(2, sleep_ms); },
        [&](){ sleep_id(3, sleep_ms); },
        [&](){ sleep_id(4, sleep_ms); },
        [&](){ sleep_id(5, sleep_ms); },
        [&](){ sleep_id(6, sleep_ms); },
        [&](){ sleep_id(7, sleep_ms); },
        [&](){ sleep_id(8, sleep_ms); },
        [&](){ sleep_id(9, sleep_ms); },
        [&](){ sleep_id(10, sleep_ms); },
    };

    return f_array;
}


static  std::vector<std::function<void()>> make_func_vector()
{
    constexpr int sleep_ms = 5;

    std::vector<std::function<void()>> f_vector;

    for (int i = 1; i <= 32; ++i)
    {
        f_vector.push_back([=](){ sleep_id(i, sleep_ms); });
    }

    return f_vector;
}


static bool execute_sequential_test()
{
    console_print("\nexecute_sequential_test\n");

    auto f_array = make_func_array();
    auto f_vector = make_func_vector();

    Stopwatch sw;

    console_print("\narray\n");
    sw.start();
    execute_sequential(f_array);
    console_print("\ntime: %f\n", sw.get_time_milli());

    console_print("\nvector\n");
    sw.start();
    execute_sequential(f_vector);
    console_print("\ntime: %f\n", sw.get_time_milli());

    return true;
}


static bool execute_parallel_test()
{
    console_print("\nexecute_parallel_test\n");

#ifndef SIMPLE_NO_PARALLEL

    auto f_array = make_func_array();
    auto f_vector = make_func_vector();

    Stopwatch sw;

    console_print("\narray\n");
    sw.start();
    execute_parallel(f_array);
    console_print("\ntime: %f\n", sw.get_time_milli());

    console_print("\nvector\n");
    sw.start();
    execute_parallel(f_vector);
    console_print("\ntime: %f\n", sw.get_time_milli());

#else

    console_print("NA\n");

#endif

    return true;
}


static bool process_range_test()
{
    console_print("\nprocess_range_test\n");

    auto const id_func = [](u32 id){ sleep_id(id, 5); };

    u32 id_begin = 0;
    u32 id_end = 32;

    Stopwatch sw;
    sw.start();
    process_range(id_begin, id_end, id_func);
    console_print("\ntime: %f\n", sw.get_time_milli());

    return true;
}


bool execute_tests()
{
    printf("\n*** execute tests ***\n");

    auto result =
        execute_sequential_test() &&
        execute_parallel_test() &&
        process_range_test();
    
    if (result)
    {
        printf("execute tests OK\n");
    }

    return result;
}