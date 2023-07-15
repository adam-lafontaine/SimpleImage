#include "../../tests_include.hpp"
#include "../../../simage/src/util/profiler.cpp"


constexpr u32 WIDTH = 2000;
constexpr u32 HEIGHT = 2000;


void run_profile_tests()
{
    constexpr u32 n_channels_32 = 32;
    constexpr u32 n_channels_8 = 6;

    auto const range = make_range(WIDTH / 2, HEIGHT / 2);
    auto const blue = img::to_pixel(32, 64, 128);

    Point2Du32 pt = { WIDTH / 2, HEIGHT / 2 };
    f32 rad = 60.0f;

    auto pixels32 = PROFILE(img::create_buffer32(WIDTH * HEIGHT * n_channels_32))
    auto pixels8 = PROFILE(img::create_buffer8(WIDTH * HEIGHT * n_channels_8))

    img::Image vette32;
    img::Image caddy32;
    img::ImageGray vette8;
    img::ImageGray caddy8;

    auto vette32_v = PROFILE(img::make_view_resized_from_file(CORVETTE_PATH, vette32, WIDTH, HEIGHT, pixels32))
    auto caddy32_v = PROFILE(img::make_view_resized_from_file(CADILLAC_PATH, caddy32, WIDTH, HEIGHT, pixels32))
    auto view32 = PROFILE(img::make_view(WIDTH, HEIGHT, pixels32))

    auto vette8_v = PROFILE(img::make_view_resized_from_file(CORVETTE_PATH, vette8, WIDTH, HEIGHT, pixels8))
    auto caddy8_v = PROFILE(img::make_view_resized_from_file(CADILLAC_PATH, caddy8, WIDTH, HEIGHT, pixels8))
    auto view8 = PROFILE(img::make_view(WIDTH, HEIGHT, pixels8))

    auto view8A = PROFILE(img::make_view(WIDTH, HEIGHT, pixels8))
    auto view8B = PROFILE(img::make_view(WIDTH, HEIGHT, pixels8))
    auto view8C = PROFILE(img::make_view(WIDTH, HEIGHT, pixels8))   

    auto sub32 = PROFILE(img::sub_view(vette32_v, range))
    auto sub8 = PROFILE(img::sub_view(vette8_v, range))

    PROFILE(img::split_rgb(vette32_v, view8A, view8B, view8C))
    PROFILE(img::split_rgba(caddy32_v, view8A, view8B, view8C, view8))
    PROFILE(img::split_hsv(vette32_v, view8A, view8B, view8C))

    PROFILE_X(img::fill(view32, blue))
    PROFILE_X(img::fill(view8, 128))

    PROFILE(img::copy(caddy32_v, view32))
    PROFILE(img::copy(caddy8_v, view8))

    PROFILE(img::map_gray(vette32_v, view8))
    PROFILE(img::map_gray(view8, view32));

    PROFILE(img::alpha_blend(caddy32_v, vette32_v, view32))

    PROFILE(img::threshold(vette8_v, view8, 128))

    PROFILE(img::blur(caddy8_v, view8))

    PROFILE(img::gradients(vette8_v, view8A))
    PROFILE(img::gradients_xy(caddy8_v, view8B, view8C))    
    
    PROFILE(img::rotate(vette32_v, view32, pt, rad))
    PROFILE(img::rotate(vette8_v, view8, pt, rad))

    //PROFILE(img::skeleton(caddy8_v, view8))

    auto viewC1a = PROFILE(img::make_view_1(WIDTH, HEIGHT, pixels32))
    auto viewC2a = PROFILE(img::make_view_2(WIDTH, HEIGHT, pixels32))
    auto viewC3a = PROFILE(img::make_view_3(WIDTH, HEIGHT, pixels32))
    auto viewC4a = PROFILE(img::make_view_4(WIDTH, HEIGHT, pixels32))
    auto viewC1b = PROFILE(img::make_view_1(WIDTH, HEIGHT, pixels32))
    auto viewC2b = PROFILE(img::make_view_2(WIDTH, HEIGHT, pixels32))
    auto viewC3b = PROFILE(img::make_view_3(WIDTH, HEIGHT, pixels32))
    auto viewC4b = PROFILE(img::make_view_4(WIDTH, HEIGHT, pixels32))

    auto subC1 = PROFILE(img::sub_view(viewC1a, range))
    auto subC2 = PROFILE(img::sub_view(viewC2a, range))
    auto subC3 = PROFILE(img::sub_view(viewC3a, range))
    auto subC4 = PROFILE(img::sub_view(viewC4a, range))

    auto ch_red = PROFILE(img::select_channel(viewC3a, img::RGB::R))
    auto ch_alpha = PROFILE(img::select_channel(viewC4a, img::RGBA::A))

    PROFILE(img::map_gray(view8, viewC1b))
    PROFILE(img::map_gray(viewC1b, view8))
    PROFILE(img::map_gray(viewC1b, view32))
    PROFILE(img::map_gray(caddy32_v, viewC1b))

    PROFILE(img::map_rgba(vette32_v, viewC4a));
    PROFILE(img::map_rgba(viewC4a, view32))

    PROFILE(img::map_rgb(caddy32_v, viewC3b))
    PROFILE(img::map_rgb(viewC3b, view32))

    PROFILE(img::map_rgb_hsv(vette32_v, viewC3a))
    PROFILE(img::map_hsv_rgb(viewC3a, view32))
    PROFILE(img::map_rgb_hsv(viewC3b, viewC3a))
    PROFILE(img::map_hsv_rgb(viewC3a, viewC3b))

    PROFILE(img::map_rgb_lch(vette32_v, viewC3a))
    PROFILE(img::map_lch_rgb(viewC3a, view32))
    PROFILE(img::map_rgb_lch(viewC3b, viewC3a))
    PROFILE(img::map_lch_rgb(viewC3a, viewC3b))

    PROFILE_X(img::fill(viewC4a, blue))
    PROFILE_X(img::fill(viewC3a, blue))
    PROFILE_X(img::fill(viewC1a, 0.5f))
    PROFILE_X(img::fill(viewC1a, (u8)255))

    PROFILE(img::threshold(viewC1b, viewC1a, 0.5f))
    PROFILE(img::threshold(viewC1b, viewC1a, 0.4f, 0.8f))

    PROFILE(img::alpha_blend(viewC4a, viewC3b, viewC3a))

    PROFILE(img::rotate(viewC1a, viewC1b, pt, rad))
    PROFILE(img::rotate(viewC2a, viewC2b, pt, rad))
    PROFILE(img::rotate(viewC3a, viewC3b, pt, rad))
    PROFILE(img::rotate(viewC4a, viewC4b, pt, rad))

    PROFILE(img::blur(viewC1a, viewC1b));
    PROFILE(img::blur(viewC3a, viewC3b));

    PROFILE(img::gradients(viewC1a, viewC1b))
    PROFILE(img::gradients_xy(viewC1a, viewC2a))

#ifndef SIMAGE_NO_CUDA

    constexpr u32 n_device_channels_32 = 1;
    constexpr u32 n_unified_channels_32 = 1;
    constexpr u32 n_device_channels_8 = 1;
    constexpr u32 n_unified_channels_8 = 1;

    img::DeviceBuffer32 dev_pixels32;
    img::DeviceBuffer32 uni_pixels32;
    img::DeviceBuffer8 dev_pixels8;
    img::DeviceBuffer8 uni_pixels8;

    auto result = PROFILE(cuda::create_device_buffer(dev_pixels32, WIDTH * HEIGHT * n_device_channels_32))
    result = PROFILE(cuda::create_device_buffer(uni_pixels32, WIDTH * HEIGHT * n_unified_channels_32))
    result = PROFILE(cuda::create_device_buffer(dev_pixels8, WIDTH * HEIGHT * n_device_channels_8))
    result = PROFILE(cuda::create_device_buffer(uni_pixels8, WIDTH * HEIGHT * n_unified_channels_8))


    PROFILE(cuda::destroy_buffer(dev_pixels32))
    PROFILE(cuda::destroy_buffer(uni_pixels32))
    PROFILE(cuda::destroy_buffer(dev_pixels8))
    PROFILE(cuda::destroy_buffer(uni_pixels8))

#endif

    PROFILE(img::destroy_buffer(pixels32))
    PROFILE(img::destroy_buffer(pixels8))
    PROFILE(img::destroy_image(vette32))
    PROFILE(img::destroy_image(caddy32))
    PROFILE(img::destroy_image(vette8))
    PROFILE(img::destroy_image(caddy8))
    
}