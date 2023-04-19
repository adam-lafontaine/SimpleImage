#include "../../tests_include.hpp"



bool init_histogram_memory(u32 width, u32 height)
{
    return true;
}


void destroy_histogram_memory()
{

}


void generate_histograms(img::View const& src, img::View const& dst)
{
    img::copy(src, dst);
}