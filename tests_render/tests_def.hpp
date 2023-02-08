#pragma once

#include "../src/simage/simage.hpp"

namespace img = simage;


void fill_platform_view_test(img::View const& out_view);

void copy_image_test(img::View const& out_view);

void resize_image_test(img::View const& out_view);

void histogram_image_test(img::View const& out);

void camera_test(img::View const& out);

void camera_callback_test(img::View const& out);

//void camera_continuous_test(img::View const& out);