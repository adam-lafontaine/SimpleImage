namespace gpuf
{
	GPU_FUNCTION
	static Point2Df32 find_rotation_src_point(Point2Du32 const& pt, Point2Du32 const& origin, f32 radians)
	{
		auto dx_dst = (f32)pt.x - (f32)origin.x;
		auto dy_dst = (f32)pt.y - (f32)origin.y;

		auto radius = hypotf(dx_dst, dy_dst);

		auto theta_dst = atan2f(dy_dst, dx_dst);
		auto theta_src = theta_dst - radians;

		auto dx_src = radius * cosf(theta_src);
		auto dy_src = radius * sinf(theta_src);

		Point2Df32 pt_src{};
		pt_src.x = (f32)origin.x + dx_src;
		pt_src.y = (f32)origin.y + dy_src;

		return pt_src;
	}


	template <typename T>
    GPU_FUNCTION
	static T* find_rotation_src_pixel(DeviceMatrix2D<T> const& view, Point2Du32 const& origin, f32 radians, u32 x, u32 y)
	{
		auto const zero = 0.0f;
		auto const width = (f32)view.width;
		auto const height = (f32)view.height;

		auto src_xy = gpuf::find_rotation_src_point({ x, y }, origin, radians);

		if (src_xy.x < zero || src_xy.x >= width || src_xy.y < zero || src_xy.y >= height)
		{
			return nullptr;
		}
		else
		{
			return gpuf::xy_at(view, __float2int_rd(src_xy.x), __float2int_rd(src_xy.y));
		}
	}	
}


namespace gpu
{
    GPU_KERNAL
    static void rotate_gray(DeviceViewGray src, DeviceViewGray dst, Point2Du32 origin, f32 radians, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

		assert(n_threads == src.width * src.height);

        auto xy = gpuf::get_thread_xy(src, t);

        auto& d = *gpuf::xy_at(dst, xy.x, xy.y);

        auto s = gpuf::find_rotation_src_pixel(src,origin, radians, xy.x, xy.y);
        
        d = s ? *s : 0;
    }


    GPU_KERNAL
    static void rotate_rgb(DeviceView src, DeviceView dst, Point2Du32 origin, f32 radians, u32 n_threads)
    {        
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

		assert(n_threads == src.width * src.height);

        auto xy = gpuf::get_thread_xy(src, t);

        auto& d = *gpuf::xy_at(dst, xy.x, xy.y);

        auto s = gpuf::find_rotation_src_pixel(src,origin, radians, xy.x, xy.y);
        
        d = s ? *s : gpuf::to_pixel(0, 0, 0);
    }

}


namespace simage
{
	void rotate(DeviceView const& src, DeviceView const& dst, Point2Du32 origin, f32 rad)
    {
        assert(verify(src, dst));

		auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::rotate_rgb, n_blocks, block_size, src, dst, origin, rad, n_threads);

		auto result = cuda::launch_success("gpu::rotate_rgb");
		assert(result);
    }


	void rotate(DeviceViewGray const& src, DeviceViewGray const& dst, Point2Du32 origin, f32 rad)
    {
        assert(verify(src, dst));

		auto const width = src.width;
		auto const height = src.height;

		auto const n_threads = width * height;
		auto const n_blocks = calc_thread_blocks(n_threads);
		constexpr auto block_size = THREADS_PER_BLOCK;

		cuda_launch_kernel(gpu::rotate_gray, n_blocks, block_size, src, dst, origin, rad, n_threads);

		auto result = cuda::launch_success("gpu::rotate_gray");
		assert(result);
    }
}
