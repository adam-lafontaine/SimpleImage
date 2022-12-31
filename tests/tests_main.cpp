#include "tests_include.hpp"


//#define LEAK_CHECK
#if defined(_WIN32) && defined(_DEBUG) && defined(LEAK_CHECK)
#include "../src/util/win32_leak_check.h"
#endif

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
        //sub_view_tests() &&
        //fill_tests() &&
        histogram_tests() &&
        true;
}

int main()
{
#if defined(_WIN32) && defined(_DEBUG) && defined(LEAK_CHECK)
    int dbgFlags = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
    dbgFlags |= _CRTDBG_CHECK_ALWAYS_DF;   // check block integrity
    dbgFlags |= _CRTDBG_DELAY_FREE_MEM_DF; // don't recycle memory
    dbgFlags |= _CRTDBG_LEAK_CHECK_DF;     // leak report on exit
    _CrtSetDbgFlag(dbgFlags);
#endif

    if (!test_success())
	{
        printf("\nTests failed\n");
		return EXIT_FAILURE;
	}

    printf("\nAll tests OK\n");
}