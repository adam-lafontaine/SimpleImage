#ifndef SIMAGE_NO_CUDA

#include "libs/cuda/device.cu"
#include "simage_types.hpp"

namespace img = simage;


template <typename T>
using DeviceMatrix2D = img::DeviceMatrix2D<T>;

using Pixel = img::Pixel;
using DeviceView = img::DeviceView;
using DeviceViewGray = img::DeviceViewGray;
using DeviceViewYUV = img::DeviceViewYUV;
using DeviceViewBGR = img::DeviceViewBGR;

using RGBA = simage::RGBA;
using RGB = simage::RGB;
using HSV = simage::HSV;
using YUV = simage::YUV;


class ChannelXY
{
public:
	u32 ch;
	u32 x;
	u32 y;
};


constexpr int THREADS_PER_BLOCK = 512;

constexpr int calc_thread_blocks(u32 n_threads)
{
    return (n_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}


/* verify */

namespace simage
{
#ifndef NDEBUG

    template <typename T>
	static bool verify(DeviceMatrix2D<T> const& view)
	{
		return view.width && view.height && view.data_;
	}


    template <class IMG_A, class IMG_B>
	static bool verify(IMG_A const& lhs, IMG_B const& rhs)
	{
		return
			verify(lhs) && verify(rhs) &&
			lhs.width == rhs.width &&
			lhs.height == rhs.height;
	}

#endif
}


namespace gpuf
{
    template <typename T>
    GPU_CONSTEXPR_FUNCTION
    inline T round_to_unsigned(f32 value)
    {
        return (T)(value + 0.5f);
    }


    GPU_CONSTEXPR_FUNCTION
    inline u8 round_to_u8(f32 value)
    {
        return gpuf::round_to_unsigned<u8>(value);
    }


    template <class VIEW>
	GPU_FUNCTION
	static Point2Du32 get_thread_xy(VIEW const& view, u32 thread_id)
	{
		// n_threads = width * height
		Point2Du32 p{};

		p.y = thread_id / view.width;
		p.x = thread_id - p.y * view.width;

		return p;
	}


    template <class VIEW>
	GPU_FUNCTION
	static ChannelXY get_thread_channel_xy(VIEW const& view, u32 thread_id)
	{
		auto width = view.width;
		auto height = view.height;

		ChannelXY cxy{};

		cxy.ch = thread_id / (width * height);
		cxy.y = (thread_id - width * height * cxy.ch) / width;
		cxy.x = (thread_id - width * height * cxy.ch) - cxy.y * width;

		return cxy;
	}


    
}


#include "src/cu/row_begin.cu"
#include "src/cu/map_channels.cu"
#include "src/cu/alpha_blend.cu"



#endif // SIMAGE_NO_CUDA