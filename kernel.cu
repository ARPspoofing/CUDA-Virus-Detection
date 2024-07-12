#include <vector>

#include "defs.h"

// Max Number of threads per block capped to 1024
#define MAX_THREADS_PER_BLOCK 512
// Min Number of threads per block clamped at 64
#define MIN_THREADS_PER_BLOCK 64
// Min Input File Size
#define MIN_INPUT_FILE_SIZE 4096
// Max Input File Size
#define MAX_INPUT_FILE_SIZE 8388608
#define MAX_CONCURRENT_KERNELS 128

__device__ size_t getStringLength(const char *str) {
    size_t len = 0;
    while (str[len] != '\0') {
        len++;
    }
    return len;
}

// Kernel function to do uint_8 to char* conversion on GPU
// Kernel function to do uint_8 to char* conversion on GPU
__global__ void convertToString(const uint8_t *file_data, size_t fileSize, char *file_data_converted_storage) {
    const char hex_chars[] = "0123456789abcdef";

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    char localData[2];

    for (int i = idx; i < fileSize; i += stride) {
        //Coalesced memory access
        localData[0] = hex_chars[file_data[i] >> 4];
        localData[1] = hex_chars[file_data[i] & 0xF];
        file_data_converted_storage[i * 2] = localData[0];
        file_data_converted_storage[i * 2 + 1] = localData[1];
    }

    // Only thread 0 does termination
    if (idx == 0) {
        file_data_converted_storage[fileSize * 2] = '\0';
    }
}

__device__ bool bruteForceMatch(const char *file_data, size_t fileSize, const char *signature, size_t sigLen) {
    bool found = false;
    for (size_t i = 0; i <= fileSize - sigLen; i++) {
        found = true;
        for (size_t j = 0; j < sigLen; j++) {
            if (file_data[i + j] != signature[j] && signature[j] != '?') {
                found = false;
                break;
            }
        }
        if (found) {
            return found;
            break;
        }
    }
    return found;
}

__global__ void matchFile(size_t file_len, char *file_data_converted_storage, char **signatures, size_t *sig_sizes, uint8_t *d_results, size_t noOfSignatures) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    size_t newFileSize = file_len * 2 + 1;  

    // Use idx and stride to increment
    for (size_t i = idx; i < noOfSignatures; i += stride) {
        // Boundary check
        size_t sizeOfSignature = sig_sizes[i];
        bool found = bruteForceMatch(file_data_converted_storage, newFileSize, signatures[i], sizeOfSignature);
        d_results[idx] = static_cast<uint8_t>(found);
    }
}

dim3 calculate_num_blocks(size_t num_of_signatures, dim3 threads_per_block) {
    int blocks = static_cast<int>(num_of_signatures / threads_per_block.x);
    if (num_of_signatures % threads_per_block.x) {
        blocks++;
    }

    return dim3(blocks, 1, 1);
}

void batchRunner(std::vector<Signature> &signatures, std::vector<InputFile> &inputs) {
  std::vector<cudaStream_t> streams{};
  streams.resize(inputs.size());

  std::vector<uint8_t *> file_bufs{};
  std::vector<char *> file_bufs_converted{};
  for (size_t i = 0; i < inputs.size(); i++) {
      cudaStreamCreate(&streams[i]);
      uint8_t *ptr = 0;
      char *ptr_converted = 0;

      check_cuda_error(cudaMalloc(&ptr, inputs[i].size));
      cudaMemcpyAsync(ptr, inputs[i].data, inputs[i].size, cudaMemcpyHostToDevice, streams[i]);
      check_cuda_error(cudaMallocManaged(&ptr_converted, inputs[i].size * 2 + 1));

      file_bufs.push_back(ptr);
      file_bufs_converted.push_back(ptr_converted);

      dim3 num_of_threads = dim3(MAX_THREADS_PER_BLOCK, 1, 1);
      int blocks = static_cast<int>((inputs[i].size + num_of_threads.x - 1) / num_of_threads.x);
      dim3 num_of_blocks = dim3(blocks, 1, 1);
      convertToString<<<num_of_blocks, num_of_threads, 0, streams[i]>>>(ptr, inputs[i].size, ptr_converted);
      check_cuda_error(cudaGetLastError());
      cudaStreamSynchronize(streams[i]);
      check_cuda_error(cudaDeviceSynchronize());
  }

  char **sig_bufs;
  size_t *sig_sizes;
  cudaMallocManaged(&sig_bufs, signatures.size() * sizeof(char*));
  cudaMallocManaged(&sig_sizes, signatures.size() * sizeof(size_t));
  size_t i = 0;
  for (i = 0; i < signatures.size(); i++) {
      char *ptr = 0;
      check_cuda_error(cudaMallocManaged(&ptr, signatures[i].size));
      cudaMemcpy(ptr, signatures[i].data, signatures[i].size, cudaMemcpyHostToDevice);
      sig_bufs[i] = ptr;
      size_t size = signatures[i].size;
      sig_sizes[i] = size;
  }

  // Allocate memory for results
  uint8_t **d_results;
  size_t inputSize = inputs.size();
  size_t signatureSize = signatures.size();
  check_cuda_error(cudaMallocManaged(&d_results, inputSize * sizeof(uint8_t*)));
  for(size_t i = 0; i < inputSize; i++) {
      check_cuda_error(cudaMallocManaged(&d_results[i], signatureSize * sizeof(uint8_t)));
      cudaMemset(d_results[i], 0, signatureSize * sizeof(uint8_t));
  }

  size_t noOfSignatures = i;

  for (size_t file_idx = 0; file_idx < inputs.size(); file_idx++) {
      dim3 num_of_threads = dim3(MAX_THREADS_PER_BLOCK, 1, 1);
      dim3 num_of_blocks = calculate_num_blocks(signatureSize, num_of_threads);

      matchFile<<<num_of_blocks, num_of_threads, 0, streams[file_idx]>>>(
          inputs[file_idx].size,
          file_bufs_converted[file_idx], // this was missing
          sig_bufs,
          sig_sizes,
          d_results[file_idx], // this was missing
          signatureSize);

      cudaError_t error = cudaGetLastError();
      if(error != cudaSuccess)
      {
          // print the CUDA error message and exit
          fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
          exit(-1);
      }

  }

  for (size_t i = 0; i < inputs.size(); i++) {
    cudaStreamSynchronize(streams[i]);
    for (size_t j = 0; j < signatureSize; j++) {
      if (d_results[i][j]) {
        fprintf(stdout, "%s: %s\n", inputs[i].name.c_str(), signatures[j].name.c_str());
      }
    }
  }

  // free the device memory, though this is not strictly necessary
  // (the CUDA driver will clean up when your program exits)
  for (auto buf : file_bufs)
      cudaFree(buf);

  for (auto buf : file_bufs_converted)
      cudaFree(buf);
  
  cudaFree(sig_bufs);
  
  for(size_t i = 0; i < inputSize; i++) {
      cudaFree(d_results[i]);
  }

  cudaFree(d_results);


  // clean up streams (again, not strictly necessary)
  for (auto &s : streams)
      cudaStreamDestroy(s);
}

void runScanner(std::vector<Signature> &signatures, std::vector<InputFile> &inputs) {
  {
      cudaDeviceProp prop;
      check_cuda_error(cudaGetDeviceProperties(&prop, 0));

      fprintf(stderr, "cuda stats:\n");
      fprintf(stderr, "  # of SMs: %d\n", prop.multiProcessorCount);
      fprintf(stderr, "  global memory: %.2f MB\n", prop.totalGlobalMem / 1024.0 / 1024.0);
      fprintf(stderr, "  shared mem per block: %zu bytes\n", prop.sharedMemPerBlock);
      fprintf(stderr, "  constant memory: %zu bytes\n", prop.totalConstMem);
  }

  // Calculating the number of batches
  size_t batches = inputs.size() / static_cast<std::size_t>(MAX_CONCURRENT_KERNELS);
  if (inputs.size() % static_cast<std::size_t>(MAX_CONCURRENT_KERNELS)) {
    ++batches;
  }

  // Processing each batch
  for (size_t batch_num = 0; batch_num < batches; ++batch_num) {
    // Determine the range of files for the current batch
    size_t batch_start = batch_num * MAX_CONCURRENT_KERNELS;
    size_t batch_end = std::min(batch_start + MAX_CONCURRENT_KERNELS, inputs.size());

    // Create a vector for the current batch of files
    std::vector<InputFile> batchedFiles(inputs.begin() + batch_start, inputs.begin() + batch_end);

    // Process the current batch of files
    batchRunner(signatures, batchedFiles);
  }
}
