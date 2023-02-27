#include "../../src/app/app.hpp"

#include <cstdio>
#include <cstdlib>


bool device_buffer_tests();

bool device_copy_tests();


constexpr auto APP_TITLE = "CUDA Tests";
constexpr auto APP_VERSION = "1.0";





static bool test_success(app::AppState& state, img::CameraUSB const& camera)
{
    return 
        device_buffer_tests() &&
        device_copy_tests() &&
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