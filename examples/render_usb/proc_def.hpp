#pragma once

#include "../../src/simage/simage_platform.hpp"

namespace img = simage;


void close_camera_procs();

bool init_camera_procs(img::CameraUSB const& camera);


void show_camera(img::View const& src, img::View const& dst);

void show_blur(img::View const& src, img::View const& dst);

void show_gray(img::View const& src, img::View const& dst);

void show_gradients(img::View const& src, img::View const& dst);

void show_gradients_red(img::View const& src, img::View const& dst);

void show_gradients_green(img::View const& src, img::View const& dst);

void show_gradients_blue(img::View const& src, img::View const& dst);

void show_camera_gray(img::ViewGray const& src, img::View const& dst);

void show_inverted_gray(img::ViewGray const& src, img::View const& dst);
