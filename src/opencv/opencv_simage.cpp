#include "../simage/simage_platform.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>


constexpr int N_CAMERAS = 1;
constexpr u32 N_FRAMES = 2;


class CameraCV
{
public:
	cv::VideoCapture capture;
	cv::Mat frames[N_FRAMES];

	u32 frame_id = 0;
};


static CameraCV g_cameras[N_CAMERAS];



namespace simage
{
	static bool grab_image(CameraCV& cam)
	{
		auto& cap = cam.capture;

		if (!cap.grab())
		{
			return false;
		}

		auto& frame = cam.frames[cam.frame_id];

		if (!cap.retrieve(frame))
		{
			return false;
		}

		

		return true;
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

		camera.id = 0;
		camera.image_width = (u32)cap.get(cv::CAP_PROP_FRAME_WIDTH);
		camera.image_height = (u32)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
		camera.max_fps = (u32)cap.get(cv::CAP_PROP_FPS);

		return true;
	}


	void close_all_cameras()
	{
		for (int i = 0; i < N_CAMERAS; ++i)
		{
			g_cameras[i].capture.release();

			for (u32 f = 0; f < N_FRAMES; ++f)
			{
				g_cameras[i].frames[f].release();
			}
		}
	}


	bool grab_image(CameraUSB const& camera, View const& dst)
	{
		if (camera.id < 0)
		{
			return false;
		}

		auto& camcv = g_cameras[camera.id];

		if (!grab_image(camcv))
		{
			return false;
		}

		auto& frame_id = camcv.frame_id;
		auto& frame = camcv.frames[frame_id];

		++frame_id;
		if (frame_id >= N_FRAMES)
		{
			frame_id = 0;
		}

		ImageBGR image;
		image.width = camera.image_width;
		image.height = camera.image_height;
		image.data_ = (BGR*)frame.data;

		auto src = make_view(image);
		map(src, dst);

		return true;
	}

}