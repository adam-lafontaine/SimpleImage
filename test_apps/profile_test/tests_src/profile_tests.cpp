#include "../../tests_include.hpp"
#include "../../../simage/src/util/profiler.cpp"


namespace test
{
    constexpr u32 WIDTH = 2000;
    constexpr u32 HEIGHT = 2000;


    static img::Buffer32 create_buffer32(u32 n_images)
    {
        PROFILE_BLOCK("create_buffer32")
        return img::create_buffer32(WIDTH * HEIGHT * n_images);
    }


    static img::View make_resized_view_from_file(fs::path const& path, img::Image& image, img::Buffer32& pixels)
    {
        PROFILE_BLOCK("make_resized_view")
        return img::make_view_resized_from_file(path, image, WIDTH, HEIGHT, pixels);
    }


    static void destroy_image(img::Image& image)
    {
        PROFILE_BLOCK("destroy_image")
        img::destroy_image(image);
    }


    static void destroy_buffer(img::Buffer32& buffer)
    {
        PROFILE_BLOCK("destroy_buffer")
        img::destroy_buffer(buffer);
    }
}


void run_profile_tests()
{
    perf::profile_init();

    u32 width = 2000;
    u32 height = 2000;

    auto pixels = test::create_buffer32(2);

    img::Image vette;
    img::Image caddy;

    auto src = test::make_resized_view_from_file(CORVETTE_PATH, vette, pixels);
    auto cur = test::make_resized_view_from_file(CADILLAC_PATH, caddy, pixels);



    

    test::destroy_image(vette);
    test::destroy_image(caddy);
    test::destroy_buffer(pixels);

    perf::profile_report();
}