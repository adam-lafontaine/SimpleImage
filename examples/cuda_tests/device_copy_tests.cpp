#include "tests_include.hpp"



bool device_copy_tests(img::Image const& src, img::View const& dst)
{
    printf("\n*** device copy tests ***\n");

    img::copy(img::make_view(src), dst);

    auto result = 
        true;

    if (result)
    {
        printf("device copy tests OK\n");
    }
    return result;
}