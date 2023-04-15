#include "../../tests_include.hpp"


void grab_rgb_test(img::CameraUSB const& camera, img::View const& out)
{
    img::grab_rgb(camera, out);
}


void grab_gray_test(img::CameraUSB const& camera, img::View const& out)
{
    auto grab_cb = [&](img::ViewGray const& frame) { img::map_gray(frame, out); };

    img::grab_gray(camera, grab_cb);
}