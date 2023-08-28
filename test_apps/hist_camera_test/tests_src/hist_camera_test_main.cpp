#include "../../app/app.hpp"
#include "../../tests_include.hpp"

#include <thread>


constexpr auto APP_TITLE = "USB Camera Histogram Test App";
constexpr auto APP_VERSION = "1.0";


bool init_histogram_memory(u32 width, u32 height);
void destroy_histogram_memory();
void generate_histograms(img::View const& src, img::View const& dst);


static void process_camera_frame(img::View const& src, app::AppState& state)
{
	auto id = !state.read_index;

    generate_histograms(src, state.screen_buffer[id]);

    state.read_index = id;

	state.check_for_stop();
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

    if (!init_histogram_memory(app_state.screen_buffer[0].width, app_state.screen_buffer[0].height))
    {
        img::close_camera(camera);
		return EXIT_FAILURE;
    }

    auto const on_frame_grab = [&](img::View const& frame) { process_camera_frame(frame, app_state); };

    auto const on_input = [&](auto const& input) {  };

	app_state.start();
	std::thread th([&]() { img::grab_rgb_continuous(camera, on_frame_grab, [&]() { return !app_state.signal_stop; }); });

	render_run(app_state, on_input);

	th.join();

    img::close_camera(camera);
    destroy_histogram_memory();
    return EXIT_SUCCESS;
}