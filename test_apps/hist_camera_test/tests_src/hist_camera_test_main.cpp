#include "../../app/app.hpp"
#include "../../tests_include.hpp"

#include <thread>


constexpr auto APP_TITLE = "USB Camera Histogram Test App";
constexpr auto APP_VERSION = "1.0";


bool init_histogram_memory(u32 width, u32 height);
void destroy_histogram_memory();
void generate_histograms(img::View const& src, img::View const& dst);


static void adjust_screen_views(img::CameraUSB& camera, img::View& app_screen)
{
	if (camera.frame_width == app_screen.width && camera.frame_height == app_screen.height)
	{
		return;
	}	

	// change camera roi if it is larger than the screen
	auto roi_camera = make_range(camera.frame_width, camera.frame_height);

	if (camera.frame_width > app_screen.width)
	{
		roi_camera.x_begin = (camera.frame_width - app_screen.width) / 2;
		roi_camera.x_end = roi_camera.x_begin + app_screen.width;
	}

	if (camera.frame_height > app_screen.height)
	{
		roi_camera.y_begin = (camera.frame_height - app_screen.height) / 2;
		roi_camera.y_end = roi_camera.y_begin + app_screen.height;
	}

	img::set_roi(camera, roi_camera);

	// screen view that fits in camera roi
	u32 x_adj_screen = 0;
	u32 y_adj_screen = 0;
	
	if (camera.frame_width < app_screen.width)
	{
		x_adj_screen = (app_screen.width - camera.frame_width) / 2;
	}
	
	if (camera.frame_height < app_screen.height)
	{
		y_adj_screen = (app_screen.height - camera.frame_height) / 2;
	}
	
	auto roi_screen = make_range(camera.frame_width, camera.frame_height);

	roi_screen.x_begin += x_adj_screen;
	roi_screen.x_end += x_adj_screen;
	roi_screen.y_begin += y_adj_screen;
	roi_screen.y_end += y_adj_screen;	

	app_screen = img::sub_view(app_screen, roi_screen);
}


static void process_view(img::View const& src, app::AppState& state)
{
	if (state.signal_stop)
	{
		return;
	}

    auto id = !state.buffer_index;

    generate_histograms(src, state.screen_buffer[id]);

    state.buffer_index = id;
}


int main()
{
    img::CameraUSB camera;
	if (!img::open_camera(camera))
	{
		return EXIT_FAILURE;
	}

    app::WindowSettings window_settings{};
	window_settings.app_title = APP_TITLE;
	window_settings.version = APP_VERSION;
	window_settings.screen_width = camera.frame_width;
	window_settings.screen_height = camera.frame_height;

	app::AppState app_state;

    if (!render_init(window_settings, app_state))
	{
        img::close_camera(camera);
		return EXIT_FAILURE;
	}

    adjust_screen_views(camera, app_state.screen_buffer[0]);
    adjust_screen_views(camera, app_state.screen_buffer[1]);

    if (!init_histogram_memory(app_state.screen_buffer[0].width, app_state.screen_buffer[0].height))
    {
        img::close_camera(camera);
		return EXIT_FAILURE;
    }

    auto const on_frame_grab = [&](img::View const& frame) { process_view(frame, app_state); };

    auto const on_input = [&](auto const& input) {  };

	std::thread th([&]() { img::grab_rgb_continuous(camera, on_frame_grab, [&]() { return !app_state.signal_stop; }); });

	render_run(app_state, on_input);

	th.join();

    img::close_camera(camera);
    destroy_histogram_memory();
    return EXIT_SUCCESS;
}