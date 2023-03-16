#include "../simage/simage_platform.hpp"
#include "../util/execute.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include <array>

namespace img = simage;


constexpr int N_CAMERAS = 1;


class DeviceCV
{
public:
	cv::VideoCapture capture;
	cv::Mat bgr_frames[2];
	img::ViewBGR bgr_views[2];

	u32 frame_curr = 0;
	u32 frame_prev = 1;
};


static DeviceCV g_devices[N_CAMERAS];


/* verify */

#ifndef NDEBUG

namespace simage
{
	template <typename T>
	static bool verify(MatrixView<T> const& view)
	{
		return view.matrix_width && view.width && view.height && view.matrix_data_;
	}


	static bool verify(CameraUSB const& camera)
	{
		return camera.image_width && camera.image_height && camera.max_fps && camera.device_id >= 0;
	}


	template <typename T>
	static bool verify(CameraUSB const& camera, MatrixView<T> const& view)
	{
		return verify(camera) && verify(view) &&
			camera.image_width == view.width &&
			camera.image_height == view.height;
	}
}

#endif


static void swap_frames(DeviceCV& device)
{
	device.frame_curr = device.frame_curr == 0 ? 1 : 0;
	device.frame_prev = device.frame_curr == 0 ? 1 : 0;
}


static void close_all_cameras()
{
	for (int i = 0; i < N_CAMERAS; ++i)
	{
		g_devices[i].capture.release();

		g_devices[i].bgr_frames[0].release();
		g_devices[i].bgr_frames[1].release();
	}
}


static bool grab_and_convert_current_frame(DeviceCV& device)
{
	auto& cap = device.capture;

	if (!cap.grab())
	{
		return false;
	}

	auto& frame = device.bgr_frames[device.frame_curr];	

	if (!cap.retrieve(frame))
	{
		return false;
	}

	auto& view = device.bgr_views[device.frame_curr];
	view.matrix_data_ = (img::BGRu8*)frame.data;

	return true;
}



namespace simage
{
	static bool camera_is_initialized(CameraUSB const& camera)
	{
		return camera.is_open
			&& camera.device_id >= 0
			&& camera.device_id < N_CAMERAS;
	}


	bool open_camera(CameraUSB& camera)
	{
		auto& device = g_devices[0];
		auto& cap = device.capture;

		cap = cv::VideoCapture(1);
		if (!cap.isOpened())
		{
			cap = cv::VideoCapture(0);
			if (!cap.isOpened())
			{
				return false;
			}
		}

		camera.device_id = 0;
		camera.image_width = (u32)cap.get(cv::CAP_PROP_FRAME_WIDTH);
		camera.image_height = (u32)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
		camera.max_fps = (u32)cap.get(cv::CAP_PROP_FPS);

		if (!create_image(camera.latest_frame, camera.image_width, camera.image_height))
		{
			cap.release();			
			return false;
		}

		camera.frame_roi = img::make_view(camera.latest_frame);

		ImageBGR bgr;
		bgr.width = camera.image_width;
		bgr.height = camera.image_height;
		bgr.data_ = (BGRu8*)(12345);

		device.bgr_views[0] = img::make_view(bgr);
		device.bgr_views[1] = img::make_view(bgr);

		camera.is_open = true;

		assert(camera.image_width);
		assert(camera.image_height);
		assert(camera.max_fps);

		return true;
	}


	void close_camera(CameraUSB& camera)
	{
		camera.is_open = false;

		if (camera.device_id < 0 || camera.device_id >= N_CAMERAS)
		{
			return;
		}
		
		destroy_image(camera.latest_frame);

		close_all_cameras();
	}


	bool grab_image(CameraUSB const& camera)
	{
		assert(verify(camera));

		if (!camera_is_initialized(camera))
		{
			return false;
		}

		auto& device = g_devices[camera.device_id];

		if (!grab_and_convert_current_frame(device))
		{
			return false;
		}

		auto roi = make_range(camera.frame_roi.width, camera.frame_roi.height);
		auto device_view = sub_view(device.bgr_views[device.frame_curr], roi);

		map_rgb(device_view, camera.frame_roi);

		swap_frames(device);

		return true;
	}


	bool grab_image(CameraUSB const& camera, View const& dst)
	{
		assert(verify(camera));

		if (!camera_is_initialized(camera))
		{
			return false;
		}

		auto& device = g_devices[camera.device_id];

		if (!grab_and_convert_current_frame(device))
		{
			return false;
		}

		auto roi = make_range(camera.frame_roi.width, camera.frame_roi.height);
		auto device_view = sub_view(device.bgr_views[device.frame_curr], roi);

		map_rgb(device_view, dst);

		swap_frames(device);

		return true;
	}


	bool grab_image(CameraUSB const& camera, view_callback const& grab_cb)
	{
		assert(verify(camera));

		if (!camera_is_initialized(camera))
		{
			return false;
		}

		auto& device = g_devices[camera.device_id];

		if (!grab_and_convert_current_frame(device))
		{
			return false;
		}

		auto roi = make_range(camera.frame_roi.width, camera.frame_roi.height);
		auto device_view = sub_view(device.bgr_views[device.frame_curr], roi);

        map_rgb(device_view, camera.frame_roi);
        grab_cb(camera.frame_roi);

		swap_frames(device);

        return true;
	}


	bool grab_continuous(CameraUSB const& camera, view_callback const& grab_cb, bool_f const& grab_condition)
	{
		assert(verify(camera));

		if (!camera_is_initialized(camera))
		{
			return false;
		}

		auto& device = g_devices[camera.device_id];

		auto roi = make_range(camera.frame_roi.width, camera.frame_roi.height);

		/*bool grab_ok[2] = { false, false };

		auto const grab_current = [&]() { grab_ok[device.frame_curr] = grab_and_convert_current_frame(device); };

		auto const process_previous = [&]() 
		{ 
			if (grab_ok[device.frame_prev]) 
			{ 
				auto& device_view = device.bgr_views[device.frame_prev];
				map(device_view, camera.frame_roi);
				grab_cb(camera.frame_roi);
			}			
		};

		std::array<std::function<void()>, 2> procs = 
		{
			grab_current, process_previous
		};*/

		while (grab_condition())
		{
			//execute(procs);
			//swap_frames(device);

			if (grab_and_convert_current_frame(device))
			{			
				auto device_view = sub_view(device.bgr_views[device.frame_curr], roi);
				map_rgb(device_view, camera.frame_roi);
				grab_cb(camera.frame_roi);
			}
		}

		return false;
	}

}