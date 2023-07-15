#include "../../tests_include.hpp"
#include "../../../simage/src/util/profiler.cpp"


constexpr u32 WIDTH = 2000;
constexpr u32 HEIGHT = 2000;


void run_tests()
{
    constexpr u32 n_channels_32 = 32;
    constexpr u32 n_channels_8 = 6;

    auto const range = make_range(WIDTH / 2, HEIGHT / 2);
    auto const blue = img::to_pixel(32, 64, 128);

    Point2Du32 pt = { WIDTH / 2, HEIGHT / 2 };
    f32 rad = 60.0f;

    auto pixels32 = img::create_buffer32(WIDTH * HEIGHT * n_channels_32);
    auto pixels8 = img::create_buffer8(WIDTH * HEIGHT * n_channels_8);

    img::Image vette32;
    img::Image caddy32;
    img::ImageGray vette8;
    img::ImageGray caddy8;

    auto vette32_v = img::make_view_resized_from_file(CORVETTE_PATH, vette32, WIDTH, HEIGHT, pixels32);
    auto caddy32_v = img::make_view_resized_from_file(CADILLAC_PATH, caddy32, WIDTH, HEIGHT, pixels32);
    auto view32 = img::make_view(WIDTH, HEIGHT, pixels32);

    auto vette8_v = img::make_view_resized_from_file(CORVETTE_PATH, vette8, WIDTH, HEIGHT, pixels8);
    auto caddy8_v = img::make_view_resized_from_file(CADILLAC_PATH, caddy8, WIDTH, HEIGHT, pixels8);
    auto view8 = img::make_view(WIDTH, HEIGHT, pixels8);

    auto view8A = img::make_view(WIDTH, HEIGHT, pixels8);
    auto view8B = img::make_view(WIDTH, HEIGHT, pixels8);
    auto view8C = img::make_view(WIDTH, HEIGHT, pixels8);

    auto viewC1a = img::make_view_1(WIDTH, HEIGHT, pixels32);
    auto viewC2a = img::make_view_2(WIDTH, HEIGHT, pixels32);
    auto viewC3a = img::make_view_3(WIDTH, HEIGHT, pixels32);
    auto viewC4a = img::make_view_4(WIDTH, HEIGHT, pixels32);
    auto viewC1b = img::make_view_1(WIDTH, HEIGHT, pixels32);
    auto viewC2b = img::make_view_2(WIDTH, HEIGHT, pixels32);
    auto viewC3b = img::make_view_3(WIDTH, HEIGHT, pixels32);
    auto viewC4b = img::make_view_4(WIDTH, HEIGHT, pixels32);


    PROFILE(img::fill(view32, blue))
    PROFILE(img::fill(view8, 128))

    PROFILE(img::fill(viewC4a, blue))
    PROFILE(img::fill(viewC3a, blue))
    PROFILE(img::fill(viewC1a, 0.5f))
    PROFILE(img::fill(viewC1a, (u8)255))


    img::destroy_buffer(pixels32);
    img::destroy_buffer(pixels8);
    img::destroy_image(vette32);
    img::destroy_image(caddy32);
    img::destroy_image(vette8);
    img::destroy_image(caddy8);
}