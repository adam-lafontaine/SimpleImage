namespace gpuf
{
	GPU_FUNCTION
	static Point2Di32 find_src_point(Point2Du32 pivot, Point2Du32 dst, f32 cos, f32 sin)
	{
		auto const dx = (f32)dst.x - (f32)pivot.x;
		auto const dy = (f32)dst.y - (f32)pivot.y;

		auto dxcos = cos * dx;
		auto dxsin = sin * dx;

		auto dycos = cos * dy;
		auto dysin = sin * dy;

		return {
			(i32)roundf(dxcos + dysin) + (i32)pivot.x,
			(i32)roundf(dycos - dxsin) + (i32)pivot.y
		};
	}
}


namespace gpu
{
    GPU_KERNAL
    static void rotate_gray(DeviceViewGray src, DeviceViewGray dst, Point2Du32 pivot, f32 cos, f32 sin, u32 n_threads)
    {
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

		assert(n_threads == src.width * src.height);

        auto dst_xy = gpuf::get_thread_xy(src, t);

		auto src_xy = gpuf::find_src_point(pivot, dst_xy, cos, sin);

		auto out = (src_xy.x < 0 || src_xy.x >= src.width || src_xy.y < 0 || src_xy.y >= src.height);

		auto& d = *gpuf::xy_at(dst, dst_xy.x, dst_xy.y);

		d = out ? 0 : *gpuf::xy_at(src, src_xy.x, src_xy.y);
    }


    GPU_KERNAL
    static void rotate_rgb(DeviceView src, DeviceView dst, Point2Du32 pivot, f32 cos, f32 sin, u32 n_threads)
    {        
        auto t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= n_threads)
		{
			return;
		}

		assert(n_threads == src.width * src.height);

        auto dst_xy = gpuf::get_thread_xy(src, t);

		auto src_xy = gpuf::find_src_point(pivot, dst_xy, cos, sin);

		auto out = (src_xy.x < 0 || src_xy.x >= src.width || src_xy.y < 0 || src_xy.y >= src.height);

		auto& d = *gpuf::xy_at(dst, dst_xy.x, dst_xy.y);

		d = out ? gpuf::to_pixel(0, 0, 0) : *gpuf::xy_at(src, src_xy.x, src_xy.y);
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

		cuda_launch_kernel(gpu::rotate_rgb, n_blocks, block_size, src, dst, origin, cosf(rad), sinf(rad), n_threads);

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

		cuda_launch_kernel(gpu::rotate_gray, n_blocks, block_size, src, dst, origin, cosf(rad), sinf(rad), n_threads);

		auto result = cuda::launch_success("gpu::rotate_gray");
		assert(result);
    }
}
