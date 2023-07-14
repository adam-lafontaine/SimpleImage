#include "../../tests_include.hpp"
#include "../../../simage/src/util/profiler.cpp"


constexpr u32 WIDTH = 2000;
constexpr u32 HEIGHT = 2000;


void run_profile_tests()
{
    constexpr u32 n_channels_32 = 13;
    constexpr u32 n_channels_8 = 6;

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

    auto range = make_range(WIDTH / 2, HEIGHT / 2);

    auto sub32 = PROFILE(img::sub_view(vette32_v, range))
    auto sub8 = PROFILE(img::sub_view(vette8_v, range))

    PROFILE(img::split_rgb(vette32_v, view8A, view8B, view8C))
    PROFILE(img::split_rgba(caddy32_v, view8A, view8B, view8C, view8))
    PROFILE(img::split_hsv(vette32_v, view8A, view8B, view8C))

    PROFILE(img::fill(view32, img::to_pixel(32, 64, 128)))
    PROFILE(img::fill(view8, 128))

    PROFILE(img::copy(caddy32_v, view32))
    PROFILE(img::copy(caddy8_v, view8))

    PROFILE(img::map_gray(vette32_v, view8))

    PROFILE(img::alpha_blend(caddy32_v, vette32_v, view32))

    PROFILE(img::threshold(vette8_v, view8, 128))

    PROFILE(img::blur(caddy8_v, view8))

    PROFILE(img::gradients(vette8_v, view8A))
    PROFILE(img::gradients_xy(caddy8_v, view8B, view8C))
    
    PROFILE(img::rotate(vette32_v, view32, { WIDTH / 2, HEIGHT / 2 }, 60))
    PROFILE(img::rotate(vette8_v, view8, { WIDTH / 2, HEIGHT / 2 }, 60))

    PROFILE(img::skeleton(caddy8_v, view8))

    auto viewC1 = PROFILE(img::make_view_1(WIDTH, HEIGHT, pixels32))
    auto viewC2 = PROFILE(img::make_view_2(WIDTH, HEIGHT, pixels32))
    auto viewC3 = PROFILE(img::make_view_3(WIDTH, HEIGHT, pixels32))
    auto viewC4 = PROFILE(img::make_view_4(WIDTH, HEIGHT, pixels32))

    

    PROFILE(img::destroy_image(vette32))
    PROFILE(img::destroy_image(caddy32))
    PROFILE(img::destroy_buffer(pixels32))
    PROFILE(img::destroy_buffer(pixels8))
}