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
        PROFILE_BLOCK("make_resized_view_32")
        return img::make_view_resized_from_file(path, image, WIDTH, HEIGHT, pixels);
    }


    static img::ViewGray make_resized_view_from_file(fs::path const& path, img::ImageGray& image, img::Buffer8& pixels)
    {
        PROFILE_BLOCK("make_resized_view_8")
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


    static img::ViewGray sub_view_8(img::ImageGray const& image, Range2Du32 const& range)
    {
        PROFILE_BLOCK("sub_view_image8")
        return img::sub_view(image, range);
    }


    static img::ViewGray sub_view_8(img::ViewGray const& view, Range2Du32 const& range)
    {
        PROFILE_BLOCK("sub_view_8")
        return img::sub_view(view, range);
    }


    static void split_rgb(img::View const& src, img::ViewGray const& red, img::ViewGray const& green, img::ViewGray const& blue)
    {
        PROFILE_BLOCK("split_rgb")
        img::split_rgb(src, red, green, blue);
    }


    static void split_rgba(img::View const& src, img::ViewGray const& red, img::ViewGray const& green, img::ViewGray const& blue, img::ViewGray const& alpha)
    {
        PROFILE_BLOCK("split_rgba")
        img::split_rgba(src, red, green, blue, alpha);
    }


    static void split_hsv(img::View const& src, img::ViewGray const& hue, img::ViewGray const& sat, img::ViewGray const& val)
    {
        PROFILE_BLOCK("split_hsv")
        img::split_hsv(src, hue, sat, val);
    }


    static void fill(img::View const& view, img::Pixel color)
    {
        PROFILE_BLOCK("fill_32")
        img::fill(view, color);
    }


    static void fill(img::ViewGray const& view, u8 gray)
    {
        PROFILE_BLOCK("fill_8");
        img::fill(view, gray);
    }


    static void copy(img::View const& src, img::View const& dst)
    {
        PROFILE_BLOCK("copy_32")
        img::copy(src, dst);
    }


    static void copy(img::ViewGray const& src, img::ViewGray const& dst)
    {
        PROFILE_BLOCK("copy_8")
        img::copy(src, dst);
    }



}


void run_profile_tests()
{
    auto pixels32 = prof::create_buffer32(3);
    auto pixels8 = prof::create_buffer8(6);

    img::Image vette32;
    img::Image caddy32;
    img::ImageGray vette8;
    img::ImageGray caddy8;

    auto vette32_v = prof::make_resized_view_from_file(CORVETTE_PATH, vette32, pixels32);
    auto caddy32_v = prof::make_resized_view_from_file(CADILLAC_PATH, caddy32, pixels32);
    auto view32 = prof::make_view_32(pixels32);

    auto vette8_v = prof::make_resized_view_from_file(CORVETTE_PATH, vette8, pixels8);
    auto caddy8_v = prof::make_resized_view_from_file(CADILLAC_PATH, caddy8, pixels8);
    auto view8 = prof::make_view_8(pixels8);

    auto view8A = prof::make_view_8(pixels8);
    auto view8B = prof::make_view_8(pixels8);
    auto view8C = prof::make_view_8(pixels8);
    
    auto sub32 = prof::sub_view_32(vette32,  make_range(vette32.width / 2, vette32.height / 2));
    sub32 = prof::sub_view_32(vette32_v,  make_range(WIDTH / 2, HEIGHT / 2));

    auto sub8 = prof::sub_view_8(vette8,  make_range(vette8.width / 2, vette8.height / 2));
    sub8 = prof::sub_view_8(vette8_v,  make_range(WIDTH / 2, HEIGHT / 2));

    prof::split_rgb(vette32_v, view8A, view8B, view8C);
    prof::split_rgba(caddy32_v, view8A, view8B, view8C, view8);
    prof::split_hsv(vette32_v, view8A, view8B, view8C);

    prof::fill(view32, img::to_pixel(32, 64, 128));
    prof::fill(view8, 128);

    prof::copy(caddy32_v, view32);
    prof::copy(caddy8_v, view8);

    prof::destroy_image(vette32);
    prof::destroy_image(caddy32);
    prof::destroy_buffer(pixels32);
    prof::destroy_buffer(pixels8);
}