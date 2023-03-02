#include "simage_cuda.cpp"
#include "../cuda/cuda_def.cuh"

#include <cuda_fp16.h>


using RGBA = simage::RGBA;
using RGB = simage::RGB;


template <typename T>
using DeviceView2D = simage::DeviceView2D<T>;

template <typename T, size_t N>
using DeviceChannelView2D = simage::DeviceChannelView2D<T, N>;


namespace gpu
{
    using r16 = __half;

    using View1r16 = DeviceView2D<r16>;

    template <size_t N>
    using ViewCHr16 = DeviceChannelView2D<r16, N>;

    using View4r16 = ViewCHr16<4>;
    using View3r16 = ViewCHr16<3>;
    using View2r16 = ViewCHr16<2>;

    using ViewRGBAr16 = View4r16;
    using ViewRGBr16 = View3r16;
}


using DeviceView = simage::DeviceView;
using DeviceViewGray = simage::DeviceViewGray;


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


static gpu::View1r16 as_device(simage::DeviceView1r16 const& host)
{
    static_assert(sizeof(gpu::r16) == sizeof(u16));

    gpu::View1r16 dst;
    dst.matrix_data_ = (gpu::r16*)host.matrix_data_;
    dst.matrix_width = host.matrix_width;
    dst.width = host.width;
    dst.height = host.height;
    dst.range = host.range;

    return dst;
}


template <size_t N>
static gpu::ViewCHr16<N> as_device(simage::DeviceViewCHr16<N> const& host)
{
    gpu::ViewCHr16<N> dst;
    dst.channel_data_ = (gpu::r16*)host.channel_data_;
    dst.channel_width_ = host.channel_width_;
    dst.width = host.width;
    dst.height = host.height;
    dst.range = host.range;

    return dst;
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
    inline gpu::r16 to_channel_r16(u8 value)
    {
        return __float2half(value / 255.0f);
    }


    GPU_CONSTEXPR_FUNCTION
    inline r32 clamp(r32 value)
    {
        if (value < 0.0f)
        {
            value = 0.0f;
        }
        else if (value > 1.0f)
        {
            value = 1.0f;
        }

        return value;
    }


    GPU_CONSTEXPR_FUNCTION
    inline u8 round_to_u8(r32 value)
    {
        return (u8)(u32)(value + 0.5f);
    }


    GPU_FUNCTION
    inline u8 to_channel_u8(gpu::r16 value)
    {
        return round_to_u8(clamp(__half2float(value)) * 255);
    }



}


/* map gray */

namespace gpu
{
    GPU_KERNAL
    static void to_float_16(DeviceViewGray src, View1r16 dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height);

        auto xy = gpuf::get_thread_xy(src, t);

        auto s = gpuf::xy_at(src, xy);
        auto d = gpuf::xy_at(dst, xy);

        *d = gpuf::to_channel_r16(*s);
    }


    GPU_KERNAL
    static void to_unsigned_8(View1r16 src, DeviceViewGray dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height);

        auto xy = gpuf::get_thread_xy(src, t);

        auto s = gpuf::xy_at(src, xy);
        auto d = gpuf::xy_at(dst, xy);

        *d = gpuf::to_channel_u8(*s);
    }
}


namespace simage
{
    void map(DeviceViewGray const& src, DeviceView1r16 const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::to_float_16, n_blocks, block_size, src, as_device(dst), n_threads);

        auto result = cuda::launch_success("gpu::to_float_16");
		assert(result);
    }


    void map(DeviceView1r16 const& src, DeviceViewGray const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::to_unsigned_8, n_blocks, block_size, as_device(src), dst, n_threads);

        auto result = cuda::launch_success("gpu::to_unsigned_8");
		assert(result);
    }
}


/* map rgb */

namespace gpu
{  
    GPU_KERNAL    
    static void to_rgba_float_16(DeviceView src, ViewRGBAr16 dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height * 4);

        auto cxy = gpuf::get_thread_channel_xy(src, n_threads);

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

        d = gpuf::to_channel_r16(s);
    }


    GPU_KERNAL    
    static void to_rgb_float_16(DeviceView src, ViewRGBr16 dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height * 3);

        auto cxy = gpuf::get_thread_channel_xy(src, n_threads);

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

        d = gpuf::to_channel_r16(s);
    }
    

    GPU_KERNAL
    static void to_rgba_unsigned_8(ViewRGBAr16 src, DeviceView dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height * 4);

        auto cxy = gpuf::get_thread_channel_xy(src, n_threads);

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
    static void to_rgb_unsigned_8(ViewRGBr16 src, DeviceView dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height * 3);

        auto cxy = gpuf::get_thread_channel_xy(src, n_threads);

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
    void map_rgba(DeviceView const& src, DeviceViewRGBAr16 const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height * 4;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::to_rgba_float_16, n_blocks, block_size, src, as_device(dst), n_threads);

        auto result = cuda::launch_success("gpu::to_rgba_float_16");
		assert(result);
    }


	void map_rgba(DeviceViewRGBAr16 const& src, DeviceView const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height * 4;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::to_rgba_unsigned_8, n_blocks, block_size, as_device(src), dst, n_threads);

        auto result = cuda::launch_success("gpu::to_rgba_unsigned_8");
		assert(result);
    }


    void map_rgb(DeviceView const& src, DeviceViewRGBr16 const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height * 3;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::to_rgb_float_16, n_blocks, block_size, src, as_device(dst), n_threads);

        auto result = cuda::launch_success("gpu::to_rgb_float_16");
		assert(result);
    }


	void map_rgb(DeviceViewRGBr16 const& src, DeviceView const& dst)
    {
        assert(verify(src, dst));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height * 3;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

        cuda_launch_kernel(gpu::to_rgb_unsigned_8, n_blocks, block_size, as_device(src), dst, n_threads);

        auto result = cuda::launch_success("gpu::to_rgb_unsigned_8");
		assert(result);
    }
}