// Shuffle intrinsics CUDA Sample
// This sample demonstrates the use of the shuffle intrinsic
// First, a simple example of a prefix sum using the shuffle to
// perform a scan operation is provided.
// Secondly, a more involved example of computing an integral image
// using the shuffle intrinsic is provided, where the shuffle
// scan operation and shuffle xor operations are used

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>

#include <helper_cuda.h>
#include <helper_functions.h>
#include <cmath>

#include <chrono>

// Scan using shfl - takes log2(n) steps
// This function demonstrates basic use of the shuffle intrinsic, __shfl_up,
// to perform a scan operation across a block.
// First, it performs a scan (prefix sum in this case) inside a warp
// Then to continue the scan operation across the block,
// each warp's sum is placed into shared memory.  A single warp
// then performs a shuffle scan on that shared memory.  The results
// are then uniformly added to each warp's threads.
// This pyramid type approach is continued by placing each block's
// final sum in global memory and prefix summing that via another kernel call,
// then uniformly adding across the input data via the uniform_add<<<>>> kernel.

void shfl_scan_test(int *data, int width, const sycl::nd_item<3> &item_ct1,
                    uint8_t *dpct_local, int *partial_sums = NULL) {
  auto sums = (int *)dpct_local;
  int id = ((item_ct1.get_group(2) * item_ct1.get_local_range(2)) +
            item_ct1.get_local_id(2));
  int lane_id = id % item_ct1.get_sub_group().get_local_range().get(0);
  // determine a warp_id within a block
  int warp_id = item_ct1.get_local_id(2) /
                item_ct1.get_sub_group().get_local_range().get(0);

  // Below is the basic structure of using a shfl instruction
  // for a scan.
  // Record "value" as a variable - we accumulate it along the way
  int value = data[id];

  // Now accumulate in log steps up the chain
  // compute sums, with another thread's value who is
  // distance delta away (i).  Note
  // those threads where the thread 'i' away would have
  // been out of bounds of the warp are unaffected.  This
  // creates the scan sum.

//#pragma unroll
  for (int i = 1; i <= width; i *= 2) {
    unsigned int mask = 0xffffffff;
    /*
    DPCT1023:5: The SYCL sub-group does not support mask options for
    dpct::shift_sub_group_right. You can specify
    "--use-experimental-features=masked-sub-group-operation" to use the
    experimental helper function to migrate __shfl_up_sync.
    */
    /*
    DPCT1096:48: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::shift_sub_group_right" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    int n =
        dpct::shift_sub_group_right(item_ct1.get_sub_group(), value, i, width);

    if (lane_id >= i) value += n;
  }

  // value now holds the scan value for the individual thread
  // next sum the largest values for each warp

  // write the sum of the warp to smem
  if (item_ct1.get_local_id(2) %
          item_ct1.get_sub_group().get_local_range().get(0) ==
      item_ct1.get_sub_group().get_local_range().get(0) - 1) {
    sums[warp_id] = value;
  
    //printf("Warp_id = %d, value = %d\n", warp_id, value);
  }

  /*
  DPCT1065:33: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  //
  // scan sum the warp sums
  // the same shfl scan operation, but performed on warp sums
  //
  if (warp_id == 0 &&
      lane_id < (item_ct1.get_local_range(2) /
                 item_ct1.get_sub_group().get_local_range().get(0))) {
    int warp_sum = sums[lane_id];

    int mask = (1 << (item_ct1.get_local_range(2) /
                      item_ct1.get_sub_group().get_local_range().get(0))) -
               1;
    for (int i = 1; i <= (item_ct1.get_local_range(2) /
                          item_ct1.get_sub_group().get_local_range().get(0));
         i *= 2) {
      /*
      DPCT1023:6: The SYCL sub-group does not support mask options for
      dpct::shift_sub_group_right. You can specify
      "--use-experimental-features=masked-sub-group-operation" to use the
      experimental helper function to migrate __shfl_up_sync.
      */
      /*
      DPCT1096:49: The right-most dimension of the work-group used in the SYCL
      kernel that calls this function may be less than "32". The function
      "dpct::shift_sub_group_right" may return an unexpected result on the CPU
      device. Modify the size of the work-group to ensure that the value of the
      right-most dimension is a multiple of "32".
      */
      int n = dpct::shift_sub_group_right(
          item_ct1.get_sub_group(), warp_sum, i,
          (item_ct1.get_local_range(2) /
           item_ct1.get_sub_group().get_local_range().get(0)));

      if (lane_id >= i) warp_sum += n;
    }

    sums[lane_id] = warp_sum;
  }
  /*
  DPCT1065:34: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */

  item_ct1.barrier();
  
  // perform a uniform add across warps in the block
  // read neighbouring warp's sum and add it to threads value
  int blockSum = 0;

  if (warp_id > 0) {
    blockSum = sums[warp_id - 1];
  }

  value += blockSum;

   //printf("data_id = %d, value = %d\n", id, value);


  // Now write out our result
  data[id] = value;

  // last thread has sum, write write out the block's sum
  if (partial_sums != NULL &&
      item_ct1.get_local_id(2) == item_ct1.get_local_range(2) - 1) {
    partial_sums[item_ct1.get_group(2)] = value;
  
   //  printf("partialsum_id = %d, value = %d\n", item_ct1.get_group(2), value);

  }

}

static unsigned int iDivUp(unsigned int dividend, unsigned int divisor) {
  return ((dividend % divisor) == 0) ? (dividend / divisor)
                                     : (dividend / divisor + 1);
}

void shuffle_simple_test(int argc, char **argv) {
  int *h_data, *h_partial_sums, *h_result;
  int *d_data, *d_partial_sums;
  const int n_elements =1024*8; //65536;
  int sz = sizeof(int) * n_elements;
  int cuda_device = 0;

  printf("Starting shfl_scan\n");

 checkCudaErrors(DPCT_CHECK_ERROR(
      h_data = (int *)sycl::malloc_host(sizeof(int) * n_elements,
                                        dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      h_result = (int *)sycl::malloc_host(sizeof(int) * n_elements,
                                          dpct::get_in_order_queue())));

  // initialize data:
  printf("Computing Simple Sum test\n");
  printf("---------------------------------------------------\n");

  printf("Initialize test data [1, 1, 1...]\n");

  for (int i = 0; i < n_elements; i++) {
    h_data[i] = 1;
  }

  int blockSize = 256;
  int gridSize = n_elements / blockSize;
  int nWarps = blockSize / 32;
  /*
  DPCT1083:8: The size of local memory in the migrated code may be different
  from the original code. Check that the allocated memory size in the migrated
  code is correct.
  */
  int shmem_sz = nWarps * sizeof(int);
  int n_partialSums = n_elements / blockSize;
  int partial_sz = n_partialSums * sizeof(int);

  printf("Scan summation for %d elements, %d partial sums\n", n_elements,
         n_elements / blockSize);

  checkCudaErrors(DPCT_CHECK_ERROR(
      d_data = (int *)sycl::malloc_device(sz, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      d_partial_sums =
          (int *)sycl::malloc_device(partial_sz, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memset(d_partial_sums, 0, partial_sz).wait()));

  checkCudaErrors(DPCT_CHECK_ERROR(
      h_partial_sums =
          (int *)sycl::malloc_host(partial_sz, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d_data, h_data, sz).wait()));

  /*
  DPCT1049:7: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
        sycl::range<1>(shmem_sz), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, gridSize) *
                              sycl::range<3>(1, 1, blockSize),
                          sycl::range<3>(1, 1, blockSize)),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
          shfl_scan_test(d_data, 32, item_ct1, dpct_local_acc_ct1.get_pointer(),
                         d_partial_sums);
        });
  });
 

   checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(h_result, d_data, sz).wait()));
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(h_partial_sums, d_partial_sums, partial_sz)
                           .wait()));

  printf("Test Sum: %d\n", h_partial_sums[n_partialSums - 1]);
 
  checkCudaErrors(
      DPCT_CHECK_ERROR(sycl::free(h_data, dpct::get_in_order_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(sycl::free(h_result, dpct::get_in_order_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(sycl::free(h_partial_sums, dpct::get_in_order_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(sycl::free(d_data, dpct::get_in_order_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(sycl::free(d_partial_sums, dpct::get_in_order_queue())));
}

int main(int argc, char *argv[]) {

  int cuda_device = 0;
  printf("Starting shfl_scan\n");
  dpct::device_info deviceProp;
  checkCudaErrors(cuda_device = dpct::dev_mgr::instance().current_device_id());

  checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_device_info(
      deviceProp, dpct::dev_mgr::instance().get_device(cuda_device))));

  printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
         /*
         DPCT1005:46: The SYCL device version is different from CUDA Compute
         Compatibility. You may need to rewrite this code.
         */
         deviceProp.get_major_version(), deviceProp.get_minor_version(),
         deviceProp.get_max_compute_units());

  // __shfl intrinsic needs SM 3.0 or higher
  /*
  DPCT1005:47: The SYCL device version is different from CUDA Compute
  Compatibility. You may need to rewrite this code.
  */
  if (deviceProp.get_major_version() < 3) {
    printf("> __shfl() intrinsic requires device SM 3.0+\n");
    printf("> Waiving test.\n");
    exit(EXIT_WAIVED);
  }
  
  shuffle_simple_test(argc, argv);

  return 0;
}

