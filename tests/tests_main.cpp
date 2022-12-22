#include "tests_include.hpp"


int main()
{
    if (!directory_files_test())
	{
		return EXIT_FAILURE;
	}

    execute_tests();

    stb_simage_tests();
}