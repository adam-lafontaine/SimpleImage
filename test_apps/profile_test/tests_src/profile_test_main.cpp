#include "../../tests_include.hpp"
#include "../../../simage/src/util/profiler.hpp"


void run_profile_tests();


static bool run_preliminary_tests()
{
	return directory_files_test();
}


int main()
{
    if (!run_preliminary_tests())
	{
		return EXIT_FAILURE;
	}

    perf::profile_init();

    for (u32 i = 0; i < 3; ++i)
    {
        run_profile_tests();
    }    

    perf::profile_report();

    return EXIT_SUCCESS;
}