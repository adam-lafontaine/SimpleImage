namespace gpuf
{
    template <typename T>
    GPU_FUNCTION
	static f32 gradient_x_at_xy(DeviceMatrix2D<T> const& view, u32 x, u32 y)
    {
        u32 const w = view.width;
		u32 const h = view.height;

        auto r = umin(y, h - y - 1);
        auto c = umin(x, w - x - 1);
        auto rc = umin(r, c);

        if (rc == 0)
        {
            return 0.0f;
        }

        switch(c)
        {
        case 1:
            return gpuf::convolve_at_xy_f32(view, x, y, GRAD_X_3x3, 3, 3);

        case 2:
            return gpuf::convolve_at_xy_f32(view, x, y, GRAD_X_3x5, 5, 3);
        
        case 3:
            return gpuf::convolve_at_xy_f32(view, x, y, GRAD_X_3x7, 7, 3);

        case 4:
            return gpuf::convolve_at_xy_f32(view, x, y, GRAD_X_3x9, 9, 3);
        
        default:
            return gpuf::convolve_at_xy_f32(view, x, y, GRAD_X_3x11, 11, 3);
        }
    }


    template <typename T>
    GPU_FUNCTION
	static f32 gradient_y_at_xy(DeviceMatrix2D<T> const& view, u32 x, u32 y)
    {
        u32 const w = view.width;
		u32 const h = view.height;

        auto r = umin(y, h - y - 1);
        auto c = umin(x, w - x - 1);
        auto rc = umin(r, c);

        if (rc == 0)
        {
            return 0.0f;
        }

        switch(r)
        {
        case 1:
            return gpuf::convolve_at_xy_f32(view, x, y, GRAD_Y_3x3, 3, 3);

        case 2:
            return gpuf::convolve_at_xy_f32(view, x, y, GRAD_Y_3x5, 3, 5);
        
        case 3:
            return gpuf::convolve_at_xy_f32(view, x, y, GRAD_Y_3x7, 3, 7);

        case 4:
            return gpuf::convolve_at_xy_f32(view, x, y, GRAD_Y_3x9, 3, 9);
        
        default:
            return gpuf::convolve_at_xy_f32(view, x, y, GRAD_Y_3x11, 3, 11);
        }
    }

}


namespace gpu
{
    GPU_KERNAL
    static void gradients(DeviceViewGray src, DeviceViewGray dst, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

		assert(n_threads == src.width * src.height);

		auto xy = gpuf::get_thread_xy(src, t);

        auto& d = *gpuf::xy_at(dst, xy.x, xy.y);

        auto grad_x = gpuf::gradient_x_at_xy(src, xy.x, xy.y);
        auto grad_y = gpuf::gradient_y_at_xy(src, xy.x, xy.y);

        d = gpuf::round_to_u8(hypotf(grad_x, grad_y));
    }


    GPU_KERNAL
    static void gradients_xy(DeviceViewGray src, DeviceViewGray dst_x, DeviceViewGray dst_y, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

        assert(n_threads == src.width * src.height);

		auto xy = gpuf::get_thread_xy(src, t);

        auto& d_x = *gpuf::xy_at(dst_x, xy.x, xy.y);
        auto& d_y = *gpuf::xy_at(dst_y, xy.x, xy.y);

        auto grad_x = gpuf::gradient_x_at_xy(src, xy.x, xy.y);
        auto grad_y = gpuf::gradient_y_at_xy(src, xy.x, xy.y);

        d_x = gpuf::round_to_u8(fabsf(grad_x));
        d_y = gpuf::round_to_u8(fabsf(grad_y));
    }
}


namespace simage
{
    void gradients(DeviceViewGray const& src, DeviceViewGray const& dst)
    {
        assert(verify(src, dst));

		auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::gradients, n_blocks, block_size, src, dst, n_threads);

		auto result = cuda::launch_success("gpu::gradients");
		assert(result);
    }


    void gradients_xy(DeviceViewGray const& src, DeviceViewGray const& dst_x, DeviceViewGray const& dst_y)
    {
        assert(verify(src, dst_x));
        assert(verify(src, dst_y));

        auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::gradients_xy, n_blocks, block_size, src, dst_x, dst_y, n_threads);

		auto result = cuda::launch_success("gpu::gradients_xy");
		assert(result);
    }
}