#include "simage_cuda.cpp"
#include "../cuda/cuda_def.cuh"

#include <cuda_fp16.h>


template <typename T>
using DeviceView2D = simage::DeviceView2D<T>;

template <typename T, size_t N>
using DeviceChannelView2D = simage::DeviceChannelView2D<T, N>;


namespace gpu
{
    using r16 = __half;

    using DeviceView1r16 = DeviceView2D<r16>;

    using DeviceView4r16 = DeviceChannelView2D<r16, 4>;
    using DeviceView3r16 = DeviceChannelView2D<r16, 3>;
    using DeviceView2r16 = DeviceChannelView2D<r16, 2>;
}


using DeviceView = simage::DeviceView;
using DeviceViewGray = simage::DeviceViewGray;


constexpr int THREADS_PER_BLOCK = 512;

constexpr int calc_thread_blocks(u32 n_threads)
{
    return (n_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}


static gpu::DeviceView1r16 as_device(simage::DeviceView2D<cuda::r16> const& host)
{
    static_assert(sizeof(gpu::r16) == sizeof(cuda::r16));

    gpu::DeviceView1r16 dst;
    dst.matrix_data_ = (gpu::r16*)host.matrix_data_;
    dst.matrix_width = host.matrix_width;
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


namespace gpu
{
    GPU_KERNAL
    static void to_float_16(DeviceViewGray src, DeviceView1r16 dst, u32 n_threads)
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

        d[t] = gpuf::to_channel_r16(s[t]);
    }


    GPU_KERNAL
    static void to_float_32(DeviceView1r16 src, DeviceViewGray dst, u32 n_threads)
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

        d[t] = gpuf::to_channel_u8(s[t]);
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

        cuda_launch_kernel(gpu::to_float_32, n_blocks, block_size, as_device(src), dst, n_threads);

        auto result = cuda::launch_success("gpu::to_float_32");
		assert(result);
    }
}