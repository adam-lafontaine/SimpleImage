#include <cstdio>
#include <cstdlib>


bool device_buffer_tests();

bool device_copy_tests();


static bool test_success()
{
    return 
        device_buffer_tests() &&
        device_copy_tests() &&
        true;
}


int main()
{
    if (!test_success())
	{
        printf("\nTests failed\n");
		return EXIT_FAILURE;
	}

    printf("\nAll tests OK\n");
    return EXIT_SUCCESS;
}