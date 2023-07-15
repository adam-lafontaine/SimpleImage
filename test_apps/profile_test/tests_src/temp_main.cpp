#include "../../tests_include.hpp"
#include "../../../simage/src/util/profiler.hpp"


void run_tests();


int main()
{
    perf::profile_init();

    for (u32 i = 0; i < 3; ++i)
    {
        run_tests();
    }    

    perf::profile_report();

    return EXIT_SUCCESS;
}