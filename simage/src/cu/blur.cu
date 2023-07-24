namespace gpuf
{   
    template <typename T>
    GPU_FUNCTION
    static void blur_at_xy(DeviceMatrix2D<T> const& src, DeviceMatrix2D<T> const& dst, u32 x, u32 y)
    {
        auto w = src.width;
        auto h = src.height;

        auto& d = *gpuf::xy_at(dst, x, y);

        auto rc = umin(umin(x, w - x - 1), umin(y, h - y - 1));

        switch (rc)
        {
        case 0:
            // copy
            d = *gpuf::xy_at(src, x, y);
            return;

        case 1:
            d = gpuf::convolve_at_xy(src, x, y, GAUSS_3x3, 3, 3);
            return;

        case 2:
            d = gpuf::convolve_at_xy(src, x, y, GAUSS_5x5, 5, 5);
            return;
        
        case 3:
            d = gpuf::convolve_at_xy(src, x, y, GAUSS_7x7, 7, 7);
            return;

        case 4:
            d = gpuf::convolve_at_xy(src, x, y, GAUSS_9x9, 9, 9);
            return;
        
        default:
            d = gpuf::convolve_at_xy(src, x, y, GAUSS_11x11, 11, 11);
            return;
        }
    }
}

namespace gpu
{
    GPU_KERNAL
    static void blur_gray(DeviceViewGray src, DeviceViewGray dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

		assert(n_threads == src.width * src.height);

		auto xy = gpuf::get_thread_xy(src, t);

        gpuf::blur_at_xy(src, dst, xy.x, xy.y);
    }


    GPU_KERNAL
    static void blur_rgb(DeviceView src, DeviceView dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

		assert(n_threads == src.width * src.height);

		auto xy = gpuf::get_thread_xy(src, t);

        gpuf::blur_at_xy(src, dst, xy.x, xy.y);
    }
}


namespace simage
{
    void blur(DeviceViewGray const& src, DeviceViewGray const& dst)
    {
        assert(verify(src, dst));

		auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::blur_gray, n_blocks, block_size, src, dst, n_threads);

		auto result = cuda::launch_success("gpu::blur_gray");
		assert(result);
    }


    void blur(DeviceView const& src, DeviceView const& dst)
    {
        assert(verify(src, dst));

		auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::blur_rgb, n_blocks, block_size, src, dst, n_threads);

		auto result = cuda::launch_success("gpu::blur_rgb");
		assert(result);
    }
}