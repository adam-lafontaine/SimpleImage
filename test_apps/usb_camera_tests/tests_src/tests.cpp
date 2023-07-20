#include "../../tests_include.hpp"

static img::Image img_vette;
static img::Image img_caddy;

static img::Buffer32 buffer32;
static img::View view32_vette;   // 1
static img::View view32_caddy;   // 1
static img::View view32_src;     // 1
static img::View3f32 view3src; // 3
static img::View3f32 view3dst; // 3

static img::Buffer8 buffer8;
static img::ViewGray view8_src; // 1
static img::ViewGray view8_dst; // 1



bool init_camera_test_memory(u32 width, u32 height)
{
    u32 const n_channels = 9;

    u32 n_elements32 = width * height * n_channels;

    buffer32 = img::create_buffer32(n_elements32);

    view32_vette = img::make_view_resized_from_file(CORVETTE_PATH, img_vette, width, height, buffer32);
    view32_caddy = img::make_view_resized_from_file(CADILLAC_PATH, img_caddy, width, height, buffer32);

    view32_src = img::make_view(width, height, buffer32);
    view3src = img::make_view_3(width, height, buffer32);
    view3dst = img::make_view_3(width, height, buffer32);

    buffer8 = img::create_buffer8(width * height * 2);

    view8_src = img::make_view(width, height, buffer8);
    view8_dst = img::make_view(width, height, buffer8);

    return buffer32.data_ && img_vette.data_ && img_caddy.data_ && buffer8.data_;
}


void destroy_camera_test_memory()
{
    img::destroy_buffer(buffer32);
    img::destroy_buffer(buffer8);
    img::destroy_image(img_vette);
    img::destroy_image(img_caddy);
}


void grab_rgb_test(img::CameraUSB const& camera, img::View const& out)
{
    img::grab_rgb(camera, out);
}


void grab_gray_test(img::CameraUSB const& camera, img::View const& out)
{
    auto const grab_cb = [&](auto const& frame) { img::map_rgba(frame, out); };

    img::grab_gray(camera, grab_cb);
}


void transform_test(img::CameraUSB const& camera, img::View const& out)
{
    auto const invert = [](img::Pixel p)
    {
        p.rgba.red = 255 - p.rgba.red;
        p.rgba.green = 255 - p.rgba.green;
        p.rgba.blue = 255 - p.rgba.blue;

        return p;
    };

    auto const grab_cb = [&](auto const& frame)
    {
        img::transform(frame, out, invert);
    };

    img::grab_rgb(camera, grab_cb);
}


void threshold_min_max_test(img::CameraUSB const& camera, img::View const& out)
{
    auto const grab_cb = [&](auto const& frame)
    {
        img::threshold(frame, frame, 30, 200);
        img::map_rgba(frame, out);
    };

    img::grab_gray(camera, grab_cb);
}


void binarize_test(img::CameraUSB const& camera, img::View const& out)
{
    auto const grab_cb = [&](auto const& frame)
    {
        img::binarize(frame, frame, [](u8 p){ return p < 150; });
        img::map_rgba(frame, out);
    };

    img::grab_gray(camera, grab_cb);
}


void alpha_blend_test(img::CameraUSB const& camera, img::View const& out)
{
    static bool first_frame = true;

    if (first_frame)
    {
        img::copy(view32_vette, view32_src);
        img::for_each_pixel(view32_src, [](img::Pixel& p) { p.rgba.alpha = 128; });
        first_frame = false;
    }    

    auto const grab_cb = [&](auto const& frame)
    {
        img::alpha_blend(view32_src, frame, out);
    };

    img::grab_rgb(camera, grab_cb);
}


void blur_rgb_test(img::CameraUSB const& camera, img::View const& out)
{
    auto const grab_cb = [&](auto const& frame)
    {
        img::map_rgb(frame, view3src);
        img::blur(view3src, view3dst);
        img::map_rgb(view3dst, out);
    };

    img::grab_rgb(camera, grab_cb);
}


void gradients_tests(img::CameraUSB const& camera, img::View const& out)
{
    static bool first_frame = true;

    if (first_frame)
    {
        img::fill(view3dst, img::to_pixel( 0, 0, 0));
        first_frame = false;
    }

    auto red = img::select_channel(view3dst, img::RGB::R);

    auto const grab_cb = [&](auto const& frame)
    {
        img::gradients(frame, view8_src);
        img::map_gray(view8_src, red);
        img::map_rgb(view3dst, out);
    };

    img::grab_gray(camera, grab_cb);
}


void rotate_test(img::CameraUSB const& camera, img::View const& out)
{
    static u8 angle_u8 = 0;

    u8 increment = 5;
    Point2Du32 center = { out.width / 2, out.height / 2 };

    auto const grab_cb = [&](auto const& frame)
    {
        auto radians = (angle_u8 / 255.0f) * 2 * 3.14159f;
        img::rotate(frame, out, center, radians);
    };

    img::grab_rgb(camera, grab_cb);

    angle_u8 += increment;
}