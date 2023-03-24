#pragma once

#include "../../src/simage/simage_platform.hpp"

namespace img = simage;


void fill_platform_view_test(img::View const& out_view);

void copy_image_test(img::View const& out_view);

void resize_image_test(img::View const& out_view);

void histogram_image_test(img::View const& out);

void camera_rgb_test(img::View const& out);

void camera_rgb_callback_test(img::View const& out);

void camera_histogram_test(img::View const& out);

void camera_rgb_continuous_test(img::View const& out);