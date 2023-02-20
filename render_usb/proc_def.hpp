#pragma once

#include "../src/simage/simage_platform.hpp"

namespace img = simage;


void close_camera_procs();

bool init_camera_procs(img::CameraUSB& camera);


void show_camera(img::View const& src, img::View const& dst);

void show_gray(img::View const& src, img::View const& dst);