#include <stdio.h>
#include <stdint.h>
#include "cuda_helpers.cuh"
#include "cuda_runtime.h"
#include <iostream>
#include <iomanip>
//compile nvcc *.cu -o test -O2 -arch=sm_xx -std=c++11
int clockRate;

__global__ 
void global_latency(
    uint32_t * my_array, 
    int array_length, 
    uint32_t iterations, 
    uint32_t * duration, 
    uint32_t *index) 
{

	uint32_t start_time, end_time;
	uint32_t j = 0; 

	__shared__ uint32_t s_tvalue[256];
	__shared__ uint32_t s_index[256];

	int k;

    // Load data into L2 cache
    // Avoids cold instruction cache miss
    for(k=0; k<256; k++)
    {
		s_index[k] = 0;
		s_tvalue[k] = 0;
	}

    // for(int iter=0; iter<iterations; iter++)
    //second round 
    for (k = 0; k < iterations*256; k++) 
    {
        // clock has an overhead of 6 (Maxwell) to 16 (Kepler) cycles
        start_time = clock();

        j = my_array[j];
        // Ensure completed memory access before invoking clock again
        // Need to measure the latency of that part
        s_index[k] = j;
        end_time = clock();

        s_tvalue[k] += end_time-start_time;

    }

	my_array[array_length] = j;
	my_array[array_length+1] = my_array[j];

    for(k=0; k<256; k++)
    {
		index[k] = s_index[k];
		duration[k] = s_tvalue[k]/iterations;
	}
}



__global__ 
void clock_time(
    uint32_t iterations,
    uint32_t * duration)
{
    __shared__ uint32_t s_tvalue[1];
    __shared__ uint32_t s_dummy[1];
    uint32_t start_time, end_time, dummy_time;
    for(uint32_t i=0; i<1; i++)
        s_tvalue[i] = 0;

    for(uint32_t i=0; i<iterations; i++)
    {
        start_time = clock();
        dummy_time = clock();
        end_time = clock();
        s_tvalue[0] += end_time-start_time;
        s_dummy[0] += dummy_time;
    }
    
    duration[0] = s_tvalue[0]/iterations;
}

__global__ 
void shared_write_time(
    uint32_t * my_array, 
    int array_length, 
    uint32_t iterations, 
    uint32_t * duration, 
    uint32_t *index) 
{

	uint32_t start_time, end_time;
	uint32_t j = 0; 

	__shared__ uint32_t s_tvalue[256];
	__shared__ uint32_t s_index[256];

	int k;

    // Load data into L2 cache
    // Avoids cold instruction cache miss
    for(k=0; k<256; k++)
    {
		s_index[k] = 0;
		s_tvalue[k] = 0;
	}

    // for(int iter=0; iter<iterations; iter++)
    //second round 
    for (k = 0; k < iterations*256; k++) 
    {
        j = my_array[j];
        j++;
        // clock has an overhead of 6 (Maxwell) to 16 (Kepler) cycles
        start_time = clock();
        s_index[k] = j;
        end_time = clock();
        s_tvalue[k] += end_time-start_time;
        j--;
    }

	my_array[array_length] = j;
	my_array[array_length+1] = my_array[j];

    for(k=0; k<256; k++)
    {
		index[k] = s_index[k];
		duration[0] += s_tvalue[k]/iterations;
    }
    duration[0] /= 256;
}



void parametric_measure_write_shared(
    int N, 
    int iterations, 
    uint64_t stride, 
    uint32_t * h_timeinfo) 
{
	cudaError_t error_id;
	
    int i;
    uint32_t * h_a;
	h_a = (uint32_t *)malloc(sizeof(uint32_t) * (N+2));
	uint32_t * d_a;
	/* allocate arrays on GPU */
	error_id = cudaMalloc ((void **) &d_a, sizeof(uint32_t) * (N+2));
	if (error_id != cudaSuccess) {
		printf("Error 1.0 is %s\n", cudaGetErrorString(error_id));
	}

   	/* initialize array elements*/
	for (i=0; i<N; i++) 
		h_a[i] = 0;
    // big stride for access pattern P5 & P6 (Data cache/L1/L2 miss)
    // and is being revisited with P2 (Data cache hit, L1 miss, L2 hit)
    // & P3 (Data cache hit, L1/L2 miss)
    for (i=0; i<50; i++)
    {
        // During iteration, those will be accessed first
        h_a[i*stride*1024*1024]     = (i+1)*stride*1024*1024;
        
        // During iteration, those will be accessed last
		h_a[i*stride*1024*1024 + 1] = (i+1)*stride*1024*1024 + 1;			
    }
    // Created data for 0,1,stride*MB, stride*MB+1, ..., 49*stride*MB, 49*stride*MB+1
    // Next to access is 50*stride*MB, 50*stride*MB+1
    // ie stride 8: Access 400*MB, 400*MB+1
    // 1568 MB entry
    // stride = 1 MB -> access pattern P4 (Data cache miss, L1 hit, L2 not visited)
    uint64_t offset = stride*49*1024*1024;
	h_a[offset + 1] = offset + 2;
	h_a[offset + 2] = offset + 3;
	h_a[offset + 3] = offset + 1;	

    // Start at offset and move in strides of one MB
    // Create strides of one MB = 256*1024*sizeof(uint32_t)
	for (i=0; i< 31; i++)
        h_a[(i + 49*stride*4)*1024*256] = (i+1 + 49*stride*4)*1024*256;
        
    // Now we can restart at 1
	h_a[(30+49*stride*4)*1024*256] = 1;
	
	h_a[N] = 0;
	h_a[N+1] = 0;
	/* copy array elements from CPU to GPU */
        error_id = cudaMemcpy(d_a, h_a, sizeof(uint32_t) * N, cudaMemcpyHostToDevice);
	if (error_id != cudaSuccess) {
		printf("Error 1.1 is %s\n", cudaGetErrorString(error_id));
	}


	uint32_t *h_index = (uint32_t *)malloc(sizeof(uint32_t)*256);
	

	uint32_t *duration;
	error_id = cudaMalloc ((void **) &duration, sizeof(uint32_t));
	if (error_id != cudaSuccess) {
		printf("Error 1.2 is %s\n", cudaGetErrorString(error_id));
	}


	uint32_t *d_index;
	error_id = cudaMalloc( (void **) &d_index, sizeof(uint32_t)*256 );
	if (error_id != cudaSuccess) {
		printf("Error 1.3 is %s\n", cudaGetErrorString(error_id));
	}


	cudaThreadSynchronize ();
	/* launch kernel*/
	dim3 Db = dim3(1);
	dim3 Dg = dim3(1,1,1);


	shared_write_time <<<Dg, Db>>>(d_a, N, iterations,  duration, d_index);

	cudaThreadSynchronize ();

	error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
		printf("Error kernel is %s\n", cudaGetErrorString(error_id));
	}

	/* copy results from GPU to CPU */
	cudaThreadSynchronize();

    error_id = cudaMemcpy((void *)h_timeinfo, (void *)duration, sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if (error_id != cudaSuccess) {
		printf("Error 2.0 is %s\n", cudaGetErrorString(error_id));
	}
        error_id = cudaMemcpy((void *)h_index, (void *)d_index, sizeof(uint32_t)*256, cudaMemcpyDeviceToHost);
	if (error_id != cudaSuccess) {
		printf("Error 2.1 is %s\n", cudaGetErrorString(error_id));
	}

	cudaThreadSynchronize ();

    std::cout << "0:" << h_timeinfo[0] << ":shared write:" << clockRate << "\n";

	/* free memory on GPU */
	cudaFree(d_a);
	cudaFree(d_index);
	cudaFree(duration);


    /*free memory on CPU */
    free(h_index);
	free(h_a);
	
}

void parametric_measure_global(
    int N, 
    int iterations, 
    uint64_t stride, 
    uint32_t * h_timeinfo) 
{
	cudaError_t error_id;
	
    int i;
    uint32_t * h_a = nullptr;
	h_a = (uint32_t *)malloc(sizeof(uint32_t) * (N+2));
	uint32_t * d_a;
	/* allocate arrays on GPU */
	error_id = cudaMalloc ((void **) &d_a, sizeof(uint32_t) * (N+2));
	if (error_id != cudaSuccess) {
		printf("Error 1.0 is %s\n", cudaGetErrorString(error_id));
	}

    /* initialize array elements*/
	for (i=0; i<N; i++) 
        h_a[i] = 0;
    // big stride for access pattern P5 & P6 (Data cache/L1/L2 miss)
    // and is being revisited with P2 (Data cache hit, L1 miss, L2 hit)
    // & P3 (Data cache hit, L1/L2 miss)
    for (i=0; i<50; i++)
    {
        // During iteration, those will be accessed first
        h_a[i*stride*1024*1024]     = (i+1)*stride*1024*1024;
        
        // During iteration, those will be accessed last
        h_a[i*stride*1024*1024 + 1] = (i+1)*stride*1024*1024 + 1;			
    }
    // Created data for 0,1,stride*MB, stride*MB+1, ..., 49*stride*MB, 49*stride*MB+1
    // Next to access is 50*stride*MB, 50*stride*MB+1
    // ie stride 8: Access 400*MB, 400*MB+1
    // 1568 MB entry
    // stride = 1 MB -> access pattern P4 (Data cache miss, L1 hit, L2 not visited)
    uint64_t offset = stride*49*1024*1024;
    h_a[offset + 1] = offset + 2;
    h_a[offset + 2] = offset + 3;
    h_a[offset + 3] = offset + 1;	

    // Start at offset and move in strides of one MB
    // Create strides of one MB = 256*1024*sizeof(uint32_t)
    for (i=0; i< 31; i++)
        h_a[(i + 49*stride*4)*1024*256] = (i+1 + 49*stride*4)*1024*256;
        
    // Now we can restart at 1
    h_a[(30+49*stride*4)*1024*256] = 1;

    h_a[N] = 0;
    h_a[N+1] = 0;
	/* copy array elements from CPU to GPU */
        error_id = cudaMemcpy(d_a, h_a, sizeof(uint32_t) * N, cudaMemcpyHostToDevice);
	if (error_id != cudaSuccess) {
		printf("Error 1.1 is %s\n", cudaGetErrorString(error_id));
	}


	uint32_t *h_index = (uint32_t *)malloc(sizeof(uint32_t)*256);
	

	uint32_t *duration;
	error_id = cudaMalloc ((void **) &duration, sizeof(uint32_t)*256);
	if (error_id != cudaSuccess) {
		printf("Error 1.2 is %s\n", cudaGetErrorString(error_id));
	}


	uint32_t *d_index;
	error_id = cudaMalloc( (void **) &d_index, sizeof(uint32_t)*256 );
	if (error_id != cudaSuccess) {
		printf("Error 1.3 is %s\n", cudaGetErrorString(error_id));
	}


	cudaThreadSynchronize ();
	/* launch kernel*/
	dim3 Db = dim3(1);
	dim3 Dg = dim3(1,1,1);


	global_latency <<<Dg, Db>>>(d_a, N, iterations,  duration, d_index);

	cudaThreadSynchronize ();

	error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
		printf("Error kernel is %s\n", cudaGetErrorString(error_id));
	}

	/* copy results from GPU to CPU */
	cudaThreadSynchronize ();



    error_id = cudaMemcpy((void *)h_timeinfo, (void *)duration, sizeof(uint32_t)*256, cudaMemcpyDeviceToHost);
	if (error_id != cudaSuccess) {
		printf("Error 2.0 is %s\n", cudaGetErrorString(error_id));
	}
        error_id = cudaMemcpy((void *)h_index, (void *)d_index, 
            sizeof(uint32_t)*256, cudaMemcpyDeviceToHost);
	if (error_id != cudaSuccess) {
		printf("Error 2.1 is %s\n", cudaGetErrorString(error_id));
	}

	cudaThreadSynchronize ();

    // print accessed index and data acess latencies
	// for(i=0;i<256;i++)
        // printf("%d:%d\n",);
        
    for(i=0; i<256; i++)
    {
        std::cout << (h_index[i]-i)/(1024*256*sizeof(uint32_t)) << ":" 
            << h_timeinfo[i] << ":global:" << clockRate << "\n";
    }
	/* free memory on GPU */
	cudaFree(d_a);                                                      CUERR
	cudaFree(d_index);                                                  CUERR 
	cudaFree(duration);                                                 CUERR


    /*free memory on CPU */
    free(h_index);
	free(h_a);
}





void parametric_measure_clock(
    int iterations,
    uint32_t * h_timeinfo)
{
    uint32_t * duration = nullptr;
    cudaMalloc(&duration, sizeof(uint32_t));                            CUERR 
    clock_time<<<1, 1>>>(iterations, duration);                         CUERR 
    cudaThreadSynchronize ();
    cudaMemcpy(h_timeinfo, duration, sizeof(uint32_t), D2H);            CUERR 
    cudaThreadSynchronize ();
    cudaFree(duration);                                                 CUERR
    std::cout << "0:" << h_timeinfo[0] << ":clock():" << clockRate << "\n";
}


void measure_global() {

    // Get clock rate in kHz
    cudaDeviceProp cuda_prop;
    cudaGetDeviceProperties(&cuda_prop, 0);                             CUERR
    clockRate = cuda_prop.clockRate;
	 
	//stride in element
    int iterations = 10;
    int iterations_kernel = 1;
    // Stride of 32 MB where we use 4 byte words (uint32_t) as array
    uint64_t stride = 8;
    // Biggest access 
    int N = 1+(30+49*stride*4)*1024*256;
    
    /* allocate arrays on CPU */
    uint32_t * h_timeinfo = nullptr;
    std::cout << std::setprecision(16) 
        << "stride:memory latency (clock cycles):function:clock rate (kHz)\n";
    cudaMallocHost(&h_timeinfo, sizeof(uint32_t)*258);                  CUERR	
    for(uint32_t i=0; i<iterations; i++)
        parametric_measure_global(N, iterations_kernel, stride, h_timeinfo);
    for(uint32_t i=0; i<iterations; i++)
        parametric_measure_write_shared(N, iterations_kernel, stride, h_timeinfo+256);
    iterations_kernel = 10000;
    for(uint32_t i=0; i<iterations; i++)
        parametric_measure_clock(iterations_kernel, &(h_timeinfo[257]));
    cudaFreeHost(h_timeinfo);                                           CUERR    
}

int main(){

	cudaSetDevice(0);

	measure_global();

	cudaDeviceReset();
	return 0;
}
