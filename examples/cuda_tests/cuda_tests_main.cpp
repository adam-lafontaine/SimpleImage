#include "../../src/app/app.hpp"

#include <cstdio>
#include <cstdlib>
#include <thread>


bool device_buffer_tests();
bool make_view_tests();

bool copy_image_test(img::View const& src, img::View const& dst);
bool copy_sub_view_test(img::View const& src, img::View const& dst);
bool copy_gray_image_test(img::View const& src, img::View const& dst);
bool copy_gray_sub_view_test(img::View const& src, img::View const& dst);
bool map_rgba_test(img::View const& src, img::View const& dst);
bool map_rgb_test(img::View const& src, img::View const& dst);
bool map_gray_test(img::View const& src, img::View const& dst);
bool map_hsv_test(img::View const& src, img::View const& dst);
bool map_hsv_red_test(img::View const& src, img::View const& dst);
bool map_hsv_green_test(img::View const& src, img::View const& dst);
bool map_hsv_blue_test(img::View const& src, img::View const& dst);
bool map_yuv_test(img::View const& src, img::View const& dst);
bool map_yuv_red_test(img::View const& src, img::View const& dst);
bool map_yuv_green_test(img::View const& src, img::View const& dst);
bool map_yuv_blue_test(img::View const& src, img::View const& dst);


constexpr auto APP_TITLE = "CUDA Tests";
constexpr auto APP_VERSION = "1.0";


static bool run_test(img::CameraUSB const& camera, app::AppState& state, std::function<bool(img::View const&, img::View const&)> const& test)
{
    if (!img::grab_image(camera))
    {
        printf("Image grab failed\n");
        return false;
    }

    auto result = test(camera.frame_roi, state.screen_pixels);
    render_once();

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    return result;
}


static bool test_success(app::AppState& state, img::CameraUSB const& camera)
{

    return 
        /*device_buffer_tests() &&
        make_view_tests() &&
        run_test(camera, state, copy_image_test) &&
        run_test(camera, state, copy_sub_view_test) &&
        run_test(camera, state, copy_gray_image_test) &&
        run_test(camera, state, copy_gray_sub_view_test) &&
        run_test(camera, state, map_rgba_test) &&
        run_test(camera, state, map_rgb_test) &&
        run_test(camera, state, map_gray_test) &&
        run_test(camera, state, map_hsv_test) &&
        run_test(camera, state, map_hsv_red_test) &&
        run_test(camera, state, map_hsv_green_test) &&
        run_test(camera, state, map_hsv_blue_test) &&*/
        run_test(camera, state, map_yuv_test) &&
        run_test(camera, state, map_yuv_red_test) &&
        run_test(camera, state, map_yuv_green_test) &&
        run_test(camera, state, map_yuv_blue_test) &&
        true;
}


int main()
{
    img::CameraUSB camera;
	if (!img::open_camera(camera))
	{
		return false;
	}

    auto const cleanup = [&]()
    {
        img::close_camera(camera);
        render_close();
    };

	app::WindowSettings window_settings{};
	window_settings.app_title = APP_TITLE;
	window_settings.version = APP_VERSION;
	window_settings.screen_width = camera.image_width;
	window_settings.screen_height = camera.image_height;

	app::AppState app_state;

	if (!render_init(window_settings, app_state))
	{
        img::close_camera(camera);
		return EXIT_FAILURE;
	}

    if (test_success(app_state, camera))
	{
        printf("\nAll tests OK\n");		
	}
    else
    {
        printf("\nTests failed\n");
    }
    
    img::close_camera(camera);
    render_close();
    
    return EXIT_SUCCESS;
}