#include "simage_cuda.cpp"
#include "../cuda/cuda_def.cuh"


using RGBA = simage::RGBA;
using RGB = simage::RGB;


template <typename T>
using DeviceView2D = simage::DeviceView2D<T>;

template <typename T, size_t N>
using DeviceChannelView2D = simage::DeviceChannelView2D<T, N>;

using DeviceView1u16 = simage::DeviceView1u16;

using DeviceView4u16 = simage::DeviceView4u16;
using DeviceView3u16 = simage::DeviceView3u16;
using DeviceView2u16 = simage::DeviceView2u16;

using DeviceView = simage::DeviceView;
using DeviceViewGray = simage::DeviceViewGray;

using DeviceViewRGBAu16 = simage::DeviceView4u16;
using DeviceViewRGBu16 = simage::DeviceView3u16;


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


/*  */

namespace gpuf
{
    template <typename T>
	GPU_CONSTEXPR_FUNCTION
	inline int id_cast(T channel)
	{
		return static_cast<int>(channel);
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


    template <typename T>
    GPU_FUNCTION
	inline T* row_begin(DeviceView2D<T> const& view, u32 y)
	{
		return view.matrix_data_ + (u64)((view.y_begin + y) * view.matrix_width + view.x_begin);
	}


    template <typename T>
    GPU_FUNCTION
	inline T* xy_at(DeviceView2D<T> const& view, u32 x, u32 y)
    {
        return gpuf::row_begin(view, y) + x;
    }


    template <typename T>
    GPU_FUNCTION
	inline T* xy_at(DeviceView2D<T> const& view, Point2Du32 const& pt)
    {
        return gpuf::row_begin(view, pt.y) + pt.x;
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


    template <typename T, size_t N>
    GPU_FUNCTION
	inline T* channel_row_begin(DeviceChannelView2D<T, N> const& view, u32 y, u32 ch)
	{
		auto offset = (size_t)((view.y_begin + y) * view.channel_width_ + view.x_begin);

		return view.channel_data_[ch] + offset;
	}


    template <typename T, size_t N>
    GPU_FUNCTION
    inline T* channel_xy_at(DeviceChannelView2D<T, N> const& view, ChannelXY const& cxy)
    {
        return channel_row_begin(view, cxy.y, cxy.ch) + cxy.x;
    }


    GPU_FUNCTION
    inline u16 to_channel_u16(u8 value)
    {
        return (u16)value * 256;
    }


    GPU_CONSTEXPR_FUNCTION
    inline u8 round_to_u8(r32 value)
    {
        return (u8)(u32)(value + 0.5f);
    }


    GPU_FUNCTION
    inline u8 to_channel_u8(u16 value)
    {
        return u8(value / 256);
    }



}


/* map gray */

namespace gpu
{
    GPU_KERNAL
    static void to_unsigned_16(DeviceViewGray src, DeviceView1u16 dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height);

        auto xy = gpuf::get_thread_xy(src, t);

        auto s = *gpuf::xy_at(src, xy);
        auto& d = *gpuf::xy_at(dst, xy);

        d = gpuf::to_channel_u16(s);
    }


    GPU_KERNAL
    static void to_unsigned_8(DeviceView1u16 src, DeviceViewGray dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height);

        auto xy = gpuf::get_thread_xy(src, t);

        auto s = *gpuf::xy_at(src, xy);
        auto& d = *gpuf::xy_at(dst, xy);

        d = gpuf::to_channel_u8(s);
    }
}


namespace simage
{
    void map_gray(DeviceViewGray const& src, DeviceView1u16 const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::to_unsigned_16, n_blocks, block_size, src, dst, n_threads);

        auto result = cuda::launch_success("gpu::to_unsigned_16");
		assert(result);
    }


    void map_gray(DeviceView1u16 const& src, DeviceViewGray const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::to_unsigned_8, n_blocks, block_size, src, dst, n_threads);

        auto result = cuda::launch_success("gpu::to_unsigned_8");
		assert(result);
    }
}


/* map rgb */

namespace gpu
{  
    GPU_KERNAL    
    static void to_rgba_16(DeviceView src, DeviceViewRGBAu16 dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height * 4);

        auto cxy = gpuf::get_thread_channel_xy(src, t);

        auto rgba = gpuf::xy_at(src, cxy.x, cxy.y)->rgba;
        u8 s = 0;
        auto& d = *gpuf::channel_xy_at(dst, cxy);

        switch(cxy.ch)
        {
        case gpuf::id_cast(RGBA::R):
            s = rgba.red;
            break;
        case gpuf::id_cast(RGBA::G):
            s = rgba.green;
            break;
        case gpuf::id_cast(RGBA::B):
            s = rgba.blue;
            break;
        case gpuf::id_cast(RGBA::A):
            s = rgba.alpha;
            break;
        default:
            return;
        }

        d = gpuf::to_channel_u16(s);
    }


    GPU_KERNAL    
    static void to_rgb_16(DeviceView src, DeviceViewRGBu16 dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height * 3);

        auto cxy = gpuf::get_thread_channel_xy(src, t);

        auto rgba = gpuf::xy_at(src, cxy.x, cxy.y)->rgba;
        u8 s = 0;
        auto& d = *gpuf::channel_xy_at(dst, cxy);

        switch(cxy.ch)
        {
        case gpuf::id_cast(RGB::R):
            s = rgba.red;
            break;
        case gpuf::id_cast(RGB::G):
            s = rgba.green;
            break;
        case gpuf::id_cast(RGB::B):
            s = rgba.blue;
            break;
        default:
            return;
        }

        d = gpuf::to_channel_u16(s);
    }
    

    GPU_KERNAL
    static void to_rgba_8(DeviceViewRGBAu16 src, DeviceView dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height * 4);

        auto cxy = gpuf::get_thread_channel_xy(src, t);

        auto s = gpuf::to_channel_u8(*gpuf::channel_xy_at(src, cxy));
        auto& d = gpuf::xy_at(dst, cxy.x, cxy.y)->rgba;

        switch(cxy.ch)
        {
        case gpuf::id_cast(RGBA::R):
            d.red = s;
            break;
        case gpuf::id_cast(RGBA::G):
            d.green = s;
            break;
        case gpuf::id_cast(RGBA::B):
            d.blue = s;
            break;
        case gpuf::id_cast(RGBA::A):
            d.alpha = s;
            break;
        default:
            return;
        }
    }


    GPU_KERNAL
    static void to_rgb_8(DeviceViewRGBu16 src, DeviceView dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height * 3);

        auto cxy = gpuf::get_thread_channel_xy(src, t);

        auto s = gpuf::to_channel_u8(*gpuf::channel_xy_at(src, cxy));
        auto& d = gpuf::xy_at(dst, cxy.x, cxy.y)->rgba;

        switch(cxy.ch)
        {
        case gpuf::id_cast(RGB::R):
            d.red = s;
            break;
        case gpuf::id_cast(RGB::G):
            d.green = s;
            break;
        case gpuf::id_cast(RGB::B):
            d.blue = s;
            break;
        default:
            return;
        }
    }
}


namespace simage
{
    void map_rgba(DeviceView const& src, DeviceViewRGBAu16 const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height * 4;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::to_rgba_16, n_blocks, block_size, src, dst, n_threads);

        auto result = cuda::launch_success("gpu::to_rgba_16");
		assert(result);
    }


	void map_rgba(DeviceViewRGBAu16 const& src, DeviceView const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height * 4;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::to_rgba_8, n_blocks, block_size, src, dst, n_threads);

        auto result = cuda::launch_success("gpu::to_rgba_8");
		assert(result);
    }


    void map_rgb(DeviceView const& src, DeviceViewRGBu16 const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height * 3;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::to_rgb_16, n_blocks, block_size, src, dst, n_threads);

        auto result = cuda::launch_success("gpu::to_rgb_float_16");
		assert(result);
    }


	void map_rgb(DeviceViewRGBu16 const& src, DeviceView const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height * 3;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::to_rgb_8, n_blocks, block_size, src, dst, n_threads);

        auto result = cuda::launch_success("gpu::to_rgb_unsigned_8");
		assert(result);
    }
}