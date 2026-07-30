#include <iostream>
#include <chrono>
#include <cuda_runtime.h>


// ============================================================
// CUDA ERROR CHECKING HELPER
// ============================================================
//
// CUDA calls can fail silently.
// This lets us catch errors immediately.
//

#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if(error != cudaSuccess) { \
        std::cerr << "CUDA Error: " \
                  << cudaGetErrorString(error) \
                  << std::endl; \
        exit(1); \
    } \
} while(0)



// ============================================================
// STAGE 1: CPU IMPLEMENTATION
// ============================================================
//
// Baseline implementation.
// Used to compare against GPU.
//

void vectorAddCPU(
    float* A,
    float* B,
    float* C,
    int N
)
{
    for(int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
}




// ============================================================
// STAGE 2: CUDA GPU KERNEL
// ============================================================
//
// Runs on GPU.
//
// Each thread calculates one element:
//
// C[i] = A[i] + B[i]
//
// ============================================================

__global__ void vectorAddGPU(
    float* A,
    float* B,
    float* C,
    int N
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;


    if(i < N)
    {
        C[i] = A[i] + B[i];
    }
}




int main()
{

    // ============================================================
    // STAGE 3: DEFINE PROBLEM SIZE
    // ============================================================

    int N = 10000000;

    size_t bytes = N * sizeof(float);


    std::cout 
        << "Vector size: "
        << N
        << "\n\n";



    // ============================================================
    // STAGE 4: ALLOCATE CPU MEMORY
    // ============================================================
    //
    // Stored in normal system RAM.
    //
    // Host:
    //
    // A
    // B
    // C_cpu
    // C_gpu
    //

    float* A = new float[N];
    float* B = new float[N];

    float* C_cpu = new float[N];
    float* C_gpu = new float[N];



    // ============================================================
    // STAGE 5: INITIALIZE DATA
    // ============================================================

    for(int i = 0; i < N; i++)
    {
        A[i] = i;
        B[i] = 2*i;
    }




    // ============================================================
    // STAGE 6: RUN CPU VERSION + TIME IT
    // ============================================================

    auto cpu_start =
        std::chrono::high_resolution_clock::now();


    vectorAddCPU(
        A,
        B,
        C_cpu,
        N
    );


    auto cpu_end =
        std::chrono::high_resolution_clock::now();



    double cpu_time =
        std::chrono::duration<double, std::milli>
        (
            cpu_end - cpu_start
        ).count();



    std::cout
        << "CPU time: "
        << cpu_time
        << " ms\n";





    // ============================================================
    // STAGE 7: ALLOCATE GPU MEMORY
    // ============================================================
    //
    // Stored in GPU VRAM.
    //
    // Device:
    //
    // d_A
    // d_B
    // d_C
    //

    float* d_A;
    float* d_B;
    float* d_C;



    CUDA_CHECK(
        cudaMalloc(&d_A, bytes)
    );


    CUDA_CHECK(
        cudaMalloc(&d_B, bytes)
    );


    CUDA_CHECK(
        cudaMalloc(&d_C, bytes)
    );





    // ============================================================
    // STAGE 8: COPY DATA CPU ---> GPU
    // ============================================================

    auto gpu_total_start =
        std::chrono::high_resolution_clock::now();



    CUDA_CHECK(
        cudaMemcpy(
            d_A,
            A,
            bytes,
            cudaMemcpyHostToDevice
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_B,
            B,
            bytes,
            cudaMemcpyHostToDevice
        )
    );





    // ============================================================
    // STAGE 9: LAUNCH CUDA KERNEL
    // ============================================================

    int threads = 256;


    int blocks =
        (N + threads - 1) / threads;



    // CUDA timing events

    cudaEvent_t start;
    cudaEvent_t stop;


    CUDA_CHECK(
        cudaEventCreate(&start)
    );


    CUDA_CHECK(
        cudaEventCreate(&stop)
    );



    CUDA_CHECK(
        cudaEventRecord(start)
    );



    vectorAddGPU<<<blocks, threads>>>(
        d_A,
        d_B,
        d_C,
        N
    );



    CUDA_CHECK(
        cudaGetLastError()
    );



    CUDA_CHECK(
        cudaEventRecord(stop)
    );



    // ============================================================
    // STAGE 10: WAIT FOR GPU
    // ============================================================

    CUDA_CHECK(
        cudaEventSynchronize(stop)
    );



    float kernel_time;


    CUDA_CHECK(
        cudaEventElapsedTime(
            &kernel_time,
            start,
            stop
        )
    );





    // ============================================================
    // STAGE 11: COPY RESULT GPU ---> CPU
    // ============================================================


    CUDA_CHECK(
        cudaMemcpy(
            C_gpu,
            d_C,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );



    auto gpu_total_end =
        std::chrono::high_resolution_clock::now();



    double gpu_total_time =
        std::chrono::duration<double, std::milli>
        (
            gpu_total_end - gpu_total_start
        ).count();





    // ============================================================
    // STAGE 12: VERIFY RESULT
    // ============================================================


    bool correct = true;


    for(int i = 0; i < N; i++)
    {
        if(C_cpu[i] != C_gpu[i])
        {
            correct = false;

            std::cout
                << "Mismatch at index "
                << i
                << std::endl;

            break;
        }
    }



    if(correct)
    {
        std::cout
            << "CUDA result correct!\n";
    }
    else
    {
        std::cout
            << "CUDA result incorrect!\n";
    }





    // ============================================================
    // PERFORMANCE RESULTS
    // ============================================================


    std::cout << "\nPerformance:\n";


    std::cout
        << "GPU kernel time: "
        << kernel_time
        << " ms\n";


    std::cout
        << "GPU total time: "
        << gpu_total_time
        << " ms\n";


    std::cout
        << "Kernel speedup: "
        << cpu_time / kernel_time
        << "x\n";


    std::cout
        << "End-to-end speedup: "
        << cpu_time / gpu_total_time
        << "x\n";





    // ============================================================
    // STAGE 13: FREE GPU MEMORY
    // ============================================================

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));



    // ============================================================
    // STAGE 14: FREE CPU MEMORY
    // ============================================================

    delete[] A;
    delete[] B;
    delete[] C_cpu;
    delete[] C_gpu;



    std::cout
        << "\nFinished successfully!\n";


    return 0;
}