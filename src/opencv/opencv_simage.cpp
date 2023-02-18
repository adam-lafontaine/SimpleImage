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

	u32 frame_curr = 0;
	u32 frame_prev = 1;
};


static DeviceCV g_cameras[N_CAMERAS];


/* verify */

#ifndef NDEBUG

namespace simage
{
	template <typename T>
	static bool verify(MatrixView<T> const& view)
	{
		return view.matrix_width && view.width && view.height && view.matrix_data;
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


static void close_all_cameras()
{
	for (int i = 0; i < N_CAMERAS; ++i)
	{
		g_cameras[i].capture.release();

		g_cameras[i].bgr_frames[0].release();
		g_cameras[i].bgr_frames[1].release();
	}
}


static bool grab_current_frame(DeviceCV& cam)
{
	auto& cap = cam.capture;

	if (!cap.grab())
	{
		return false;
	}

	auto& frame = cam.bgr_frames[cam.frame_curr];

	if (!cap.retrieve(frame))
	{
		return false;
	}

	return true;
}


static void swap_frames(DeviceCV& device)
{
	device.frame_curr = device.frame_curr == 0 ? 1 : 0;
	device.frame_prev = device.frame_curr == 0 ? 1 : 0;
}



namespace simage
{
	


	static void process_previous_frame(CameraUSB const& camera, DeviceCV& device, view_callback const& grab_cb)
	{
		auto& frame = device.bgr_frames[device.frame_prev];

		ImageBGR bgr;
		bgr.width = camera.image_width;
		bgr.height = camera.image_height;
		bgr.data_ = (BGRu8*)frame.data;

		auto frame_view = make_view(camera.latest_frame);
		map(make_view(bgr), frame_view);
		grab_cb(frame_view);
	}


	static void process_current_frame(CameraUSB const& camera, DeviceCV& device, view_callback const& grab_cb)
	{
		auto& frame = device.bgr_frames[device.frame_curr];

		ImageBGR bgr;
		bgr.width = camera.image_width;
		bgr.height = camera.image_height;
		bgr.data_ = (BGRu8*)frame.data;

		auto frame_view = make_view(camera.latest_frame);
		map(make_view(bgr), frame_view);
		grab_cb(frame_view);
	}


	bool open_camera(CameraUSB& camera)
	{
		auto& cap = g_cameras[0].capture;

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

		if (!camera.is_open || camera.device_id < 0 || camera.device_id >= N_CAMERAS)
		{
			return false;
		}

		auto& device = g_cameras[camera.device_id];

		if (!grab_current_frame(device))
		{
			return false;
		}

		auto& frame = device.bgr_frames[device.frame_curr];

		swap_frames(device);

		ImageBGR bgr;
		bgr.width = camera.image_width;
		bgr.height = camera.image_height;
		bgr.data_ = (BGRu8*)frame.data;

		map(make_view(bgr), make_view(camera.latest_frame));

		return true;
	}


	bool grab_image(CameraUSB const& camera, View const& dst)
	{
		assert(verify(camera, dst));

		if (!camera.is_open || camera.device_id < 0 || camera.device_id >= N_CAMERAS)
		{
			return false;
		}

		auto& device = g_cameras[camera.device_id];

		if (!grab_current_frame(device))
		{
			return false;
		}

		auto& frame = device.bgr_frames[device.frame_curr];

		swap_frames(device);

		ImageBGR bgr;
		bgr.width = camera.image_width;
		bgr.height = camera.image_height;
		bgr.data_ = (BGRu8*)frame.data;

		map(make_view(bgr), dst);

		return true;
	}


	bool grab_image(CameraUSB const& camera, view_callback const& grab_cb)
	{
		if (!camera.is_open || camera.device_id < 0 || camera.device_id >= N_CAMERAS)
		{
			return false;
		}

		auto& device = g_cameras[camera.device_id];

		if (!grab_current_frame(device))
		{
			return false;
		}

		auto& frame = device.bgr_frames[device.frame_curr];

		swap_frames(device);

		ImageBGR bgr;
		bgr.width = camera.image_width;
		bgr.height = camera.image_height;
		bgr.data_ = (BGRu8*)frame.data;

		auto frame_view = make_view(camera.latest_frame);
        map(make_view(bgr), frame_view);
        grab_cb(frame_view);

        return true;
	}


	bool grab_continuous(CameraUSB const& camera, view_callback const& grab_cb, bool_f const& grab_condition)
	{
		if (!camera.is_open || camera.device_id < 0 || camera.device_id >= N_CAMERAS)
		{
			return false;
		}		

		auto& device = g_cameras[camera.device_id];
		bool grab_ok[2] = { false, false };

		auto const grab_current = [&]() { grab_ok[device.frame_curr] = grab_current_frame(device); };

		auto const process_previous = [&]() 
		{ 
			if (grab_ok[device.frame_prev]) 
			{ 
				process_previous_frame(camera, device, grab_cb);
			}			
		};

		std::array<std::function<void()>, 2> procs = 
		{
			grab_current, process_previous
		};

		while (grab_condition())
		{
			execute(procs);
			swap_frames(device);

			/*if (grab_current_frame(device))
			{
				process_current_frame(camera, device, grab_cb);
			}*/
		}

		return false;
	}

}