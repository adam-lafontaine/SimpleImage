#include "simage.hpp"
#include "src/util/execute.cpp"
#include "src/util/color_space.hpp"

#include <cmath>

namespace cs = color_space;


static void process_by_row(u32 n_rows, id_func_t const& row_func)
{
	auto const row_begin = 0;
	auto const row_end = n_rows;

	process_range(row_begin, row_end, row_func);
}


#include "src/impl/verify.cpp"
#include "src/impl/channel_pixels.cpp"
#include "src/impl/platform_image.cpp"
#include "src/impl/row_begin.cpp"
#include "src/impl/make_view.cpp"
#include "src/impl/sub_view.cpp"
#include "src/impl/fill.cpp"
#include "src/impl/convolve.cpp"
#include "src/impl/gradients.cpp"
#include "src/impl/blur.cpp"
#include "src/impl/rotate.cpp"
#include "src/impl/split_channels.cpp"
#include "src/impl/copy.cpp"
#include "src/impl/map_channels.cpp"
#include "src/impl/alpha_blend.cpp"
#include "src/impl/for_each_pixel.cpp"
#include "src/impl/transform.cpp"
#include "src/impl/centroid.cpp"
#include "src/impl/skeleton.cpp"
#include "src/impl/histogram.cpp"
#include "src/impl/select_channel.cpp"

#include "src/util/profiler.cpp"

#include "src/simage_impl.cpp"

#include "libs/stb/stb_simage.cpp"

#ifndef SIMAGE_NO_USB_CAMERA

#ifdef _WIN32

#include "libs/opencv/opencv_simage.cpp"

#else

#include "libs/uvc/uvc_simage.cpp"

#endif // _WIN32

#endif // !SIMAGE_NO_USB_CAMERA