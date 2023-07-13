#include "../../tests_include.hpp"
#include "../../../simage/src/util/profiler.cpp"


constexpr u32 WIDTH = 2000;
constexpr u32 HEIGHT = 2000;


namespace prof
{
    static img::Buffer32 create_buffer32(u32 n_images)
    {
        PROFILE_BLOCK("create_buffer32")
        return img::create_buffer32(WIDTH * HEIGHT * n_images);
    }

    static img::Buffer8 create_buffer8(u32 n_images)
    {
        PROFILE_BLOCK("create_buffer8")
        return img::create_buffer8(WIDTH * HEIGHT * n_images);
    }


    static void destroy_buffer(img::Buffer32& buffer)
    {
        PROFILE_BLOCK("destroy_buffer32")
        img::destroy_buffer(buffer);
    }


    static void destroy_buffer(img::Buffer8& buffer)
    {
        PROFILE_BLOCK("destroy_buffer8")
        img::destroy_buffer(buffer);
    }


    static img::View make_resized_view_from_file(fs::path const& path, img::Image& image, img::Buffer32& pixels)
    {
        PROFILE_BLOCK("make_resized_view")
        return img::make_view_resized_from_file(path, image, WIDTH, HEIGHT, pixels);
    }
    
    
    static img::View make_view_32(img::Buffer32& buffer)
    {
        PROFILE_BLOCK("make_view_32")
        return img::make_view(WIDTH, HEIGHT, buffer);
    }


    static img::ViewGray make_view_8(img::Buffer8& buffer)
    {
        PROFILE_BLOCK("make_view_8")
        return img::make_view(WIDTH, HEIGHT, buffer);
    }


    static void destroy_image(img::Image& image)
    {
        PROFILE_BLOCK("destroy_image")
        img::destroy_image(image);
    }


    static img::View sub_view_32(img::Image const& image, Range2Du32 const& range)
    {
        PROFILE_BLOCK("sub_view_image32")
        return img::sub_view(image, range);
    }


    static img::View sub_view_32(img::View const& view, Range2Du32 const& range)
    {
        PROFILE_BLOCK("sub_view_32")
        return img::sub_view(view, range);
    }



}


void run_profile_tests()
{
    auto pixels = prof::create_buffer32(3);
    auto gray_pixels = prof::create_buffer8(3);

    img::Image vette;
    img::Image caddy;

    auto vette_v = prof::make_resized_view_from_file(CORVETTE_PATH, vette, pixels);
    auto caddy_v = prof::make_resized_view_from_file(CADILLAC_PATH, caddy, pixels);
    auto view32 = prof::make_view_32(pixels);

    auto view8A = prof::make_view_8(gray_pixels);
    auto view8B = prof::make_view_8(gray_pixels);
    auto view8C = prof::make_view_8(gray_pixels);
    
    auto sub32 = prof::sub_view_32(vette,  make_range(vette.width / 2, vette.height / 2));
    sub32 = prof::sub_view_32(vette_v,  make_range(WIDTH / 2, HEIGHT / 2));





    

    prof::destroy_image(vette);
    prof::destroy_image(caddy);
    prof::destroy_buffer(pixels);
    prof::destroy_buffer(gray_pixels);
}