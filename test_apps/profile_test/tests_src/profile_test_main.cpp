#include "../../tests_include.hpp"



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

    run_profile_tests();

    return EXIT_SUCCESS;
}