#include "tests_include.hpp"

static bool test_success()
{
    return 
        //directory_files_test() &&
        //execute_tests() &&
        //memory_buffer_tests() &&
        //stb_simage_tests() &&
        //create_image_tests() &&
        //make_view_tests() &&
        //map_tests() &&
        //map_rgb_tests() &&
        map_rgb_hsv_tests() &&
        map_rgb_yuv_tests() &&
        true;
}

int main()
{
    if (!test_success())
	{
		return EXIT_FAILURE;
	}

    printf("\nAll tests OK\n");
}