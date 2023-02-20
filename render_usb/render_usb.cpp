#include "proc_def.hpp"
#include "../src/simage/simage.hpp"


void show_camera(img::View const& src, img::View const& dst)
{
	img::copy(src, dst);
}