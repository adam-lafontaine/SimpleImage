#pragma once

#include "../../src/simage/simage_cuda.hpp"

#include <cstdio>

namespace img = simage;


inline void fill_green(img::View const& view)
{
    img::fill(view, img::to_pixel(0, 255, 0));
}