// OpenCL Utility Functions
#include "ocl_util.h"

#ifdef _WIN32 // Windows
#include <windows.h>
#include <malloc.h>
#else         // Linux
#include <stdio.h> 
#include <unistd.h>
#endif

#ifdef __MINGW32__
#define _aligned_malloc __mingw_aligned_malloc
#define _aligned_free  __mingw_aligned_free
#endif //MINGW


// This funtion is defined in main()
extern void cleanup();

namespace ocl_util {

// Print the error associciated with an error code
void printError(cl_int error) {
	// Print error message
	switch(error)
	{
		case -1:
			printf("CL_DEVICE_NOT_FOUND ");
			break;
		case -2:
			printf("CL_DEVICE_NOT_AVAILABLE ");
			break;
		case -3:
			printf("CL_COMPILER_NOT_AVAILABLE ");
			break;
		case -4:
			printf("CL_MEM_OBJECT_ALLOCATION_FAILURE ");
			break;
		case -5:
			printf("CL_OUT_OF_RESOURCES ");
			break;
		case -6:
			printf("CL_OUT_OF_HOST_MEMORY ");
			break;
		case -7:
			printf("CL_PROFILING_INFO_NOT_AVAILABLE ");
			break;
		case -8:
			printf("CL_MEM_COPY_OVERLAP ");
			break;
		case -9:
			printf("CL_IMAGE_FORMAT_MISMATCH ");
			break;
		case -10:
			printf("CL_IMAGE_FORMAT_NOT_SUPPORTED ");
			break;
		case -11:
			printf("CL_BUILD_PROGRAM_FAILURE ");
			break;
		case -12:
			printf("CL_MAP_FAILURE ");
			break;
		case -13:
			printf("CL_MISALIGNED_SUB_BUFFER_OFFSET ");
			break;
		case -14:
			printf("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST ");
			break;

		case -30:
			printf("CL_INVALID_VALUE ");
			break;
		case -31:
			printf("CL_INVALID_DEVICE_TYPE ");
			break;
		case -32:
			printf("CL_INVALID_PLATFORM ");
			break;
		case -33:
			printf("CL_INVALID_DEVICE ");
			break;
		case -34:
			printf("CL_INVALID_CONTEXT ");
			break;
		case -35:
			printf("CL_INVALID_QUEUE_PROPERTIES ");
			break;
		case -36:
			printf("CL_INVALID_COMMAND_QUEUE ");
			break;
		case -37:
			printf("CL_INVALID_HOST_PTR ");
			break;
		case -38:
			printf("CL_INVALID_MEM_OBJECT ");
			break;
		case -39:
			printf("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR ");
			break;
		case -40:
			printf("CL_INVALID_IMAGE_SIZE ");
			break;
		case -41:
			printf("CL_INVALID_SAMPLER ");
			break;
		case -42:
			printf("CL_INVALID_BINARY ");
			break;
		case -43:
			printf("CL_INVALID_BUILD_OPTIONS ");
			break;
		case -44:
			printf("CL_INVALID_PROGRAM ");
			break;
		case -45:
			printf("CL_INVALID_PROGRAM_EXECUTABLE ");
			break;
		case -46:
			printf("CL_INVALID_KERNEL_NAME ");
			break;
		case -47:
			printf("CL_INVALID_KERNEL_DEFINITION ");
			break;
		case -48:
			printf("CL_INVALID_KERNEL ");
			break;
		case -49:
			printf("CL_INVALID_ARG_INDEX ");
			break;
		case -50:
			printf("CL_INVALID_ARG_VALUE ");
			break;
		case -51:
			printf("CL_INVALID_ARG_SIZE ");
			break;
		case -52:
			printf("CL_INVALID_KERNEL_ARGS ");
			break;
		case -53:
			printf("CL_INVALID_WORK_DIMENSION ");
			break;
		case -54:
			printf("CL_INVALID_WORK_GROUP_SIZE ");
			break;
		case -55:
			printf("CL_INVALID_WORK_ITEM_SIZE ");
			break;
		case -56:
			printf("CL_INVALID_GLOBAL_OFFSET ");
			break;
		case -57:
			printf("CL_INVALID_EVENT_WAIT_LIST ");
			break;
		case -58:
			printf("CL_INVALID_EVENT ");
			break;
		case -59:
			printf("CL_INVALID_OPERATION ");
			break;
		case -60:
			printf("CL_INVALID_GL_OBJECT ");
			break;
		case -61:
			printf("CL_INVALID_BUFFER_SIZE ");
			break;
		case -62:
			printf("CL_INVALID_MIP_LEVEL ");
			break;
		case -63:
			printf("CL_INVALID_GLOBAL_WORK_SIZE ");
			break;
		default:
			printf("UNRECOGNIZED ERROR CODE (%d)", error);
	}
}

// Print line, file name, and error code if there is an error. Exits the
// application upon error.
void _checkError(int line,
				const char *file,
				cl_int error,
                const char *msg,
                 ...) {
	// If not successful
	if(error != CL_SUCCESS) {
	// Print line and file
    printf("ERROR: ");
    printError(error);
    printf("\nLocation: %s:%d\n", file, line);

    // Print custom message.
    va_list vl;
    va_start(vl, msg);
    vprintf(msg, vl);
    printf("\n");
    va_end(vl);

    // Cleanup and bail.
    cleanup();
    exit(error);
	}
}


bool fileExists(const char *file_name) {
#ifdef _WIN32 // Windows
	DWORD attrib = GetFileAttributesA(file_name);
	return (attrib != INVALID_FILE_ATTRIBUTES && !(attrib & FILE_ATTRIBUTE_DIRECTORY));
#else         // Linux
	return access(file_name, R_OK) != -1;
#endif
}

// Load the opencl source or binary files into memory
size_t load_file_to_memory(const char *filename, char **result)
{ 
  size_t size = 0;
  FILE *f;
  
#ifdef _WIN32
  if (fopen_s(&f, filename, "rb") != 0) {
	  *result = NULL;
	  printf("Error: Could not open binary file!!!");
	  return -1; // -1 means file opening fail 
  }
#else
  f = fopen(filename, "rb");
  if (f == NULL)
  {
	  *result = NULL;
	  printf("Error: Could not open binary file!!!");
	  return -1; // -1 means file opening fail 
  }
#endif
  
  fseek(f, 0, SEEK_END);
  size = ftell(f);
  fseek(f, 0, SEEK_SET);
  *result = (char *)malloc(size+1);
  if (size != fread(*result, sizeof(char), size, f)) 
  { 
    free(*result);
	printf("Error: Could not allocate memory for binary file");
    return -2; // -2 means file reading fail 
  }
  fclose(f);
  (*result)[size] = 0; //terminating with zero
  return size;
}

////////////////////////////////////////////////
// High level funtions for platform operations 
////////////////////////////////////////////////
// Searches all platforms for the first platform whose name
// contains the search string (case-insensitive).
cl_platform_id findPlatform(const char *platform_name_search) {
  cl_int status;

  std::string search = platform_name_search;
  std::transform(search.begin(), search.end(), search.begin(), tolower);

  // Get number of platforms.
  cl_uint num_platforms;
  status = clGetPlatformIDs(0, NULL, &num_platforms);
  checkError(status, "Query for number of platforms failed");

  // Get a list of all platform ids.
  scoped_array<cl_platform_id> pids(num_platforms);
  status = clGetPlatformIDs(num_platforms, pids, NULL);
  checkError(status, "Query for all platform ids failed");

  // For each platform, get name and compare against the search string.
  for(unsigned i = 0; i < num_platforms; ++i) {
    std::string name = getPlatformName(pids[i]);

    // Convert to lower case.
    std::transform(name.begin(), name.end(), name.begin(), tolower);

    if(name.find(search) != std::string::npos) {
      // Found!
      return pids[i];
    }
  }

  // No platform found.
  return NULL;
}

// Returns the platform name
std::string getPlatformName(cl_platform_id pid) {
  cl_int status;

  size_t sz;
  status = clGetPlatformInfo(pid, CL_PLATFORM_NAME, 0, NULL, &sz);
  checkError(status, "Query for platform name size failed");

  scoped_array<char> name(sz);
  status = clGetPlatformInfo(pid, CL_PLATFORM_NAME, sz, name, NULL);
  checkError(status, "Query for platform name failed");

  return name.get();
}

// Returns the device name
std::string getDeviceName(cl_device_id did) {
  cl_int status;

  size_t sz;
  status = clGetDeviceInfo(did, CL_DEVICE_NAME, 0, NULL, &sz);
  checkError(status, "Failed to get device name size");

  scoped_array<char> name(sz);
  status = clGetDeviceInfo(did, CL_DEVICE_NAME, sz, name, NULL);
  checkError(status, "Failed to get device name");

  return name.get();
}

// Returns the list of all devices
cl_device_id *getDevices(cl_platform_id pid, cl_device_type dev_type, cl_uint *num_devices) {
  cl_int status;

  status = clGetDeviceIDs(pid, dev_type, 0, NULL, num_devices);
  checkError(status, "Query for number of devices failed");

  cl_device_id *dids = new cl_device_id[*num_devices];
  status = clGetDeviceIDs(pid, dev_type, *num_devices, dids, NULL);
  checkError(status, "Query for device ids");

  return dids;
}

// Returns the device information
void displayDeviceInfo(cl_device_id did) {
  cl_int status;

  size_t max_work_group_size;
  cl_ulong global_mem_size;
  cl_ulong local_mem_size;
  cl_uint max_clk_freq;
  cl_uint max_cu_num;

  size_t sz;
  status = clGetDeviceInfo(did,  CL_DEVICE_VERSION , 0, NULL, &sz);
  checkError(status, "Failed to get device version size");
  
  scoped_array<char> version(sz);
  status = clGetDeviceInfo(did, CL_DEVICE_VERSION, sz, version, NULL);
  checkError(status, "Failed to get device version");
  
  status = clGetDeviceInfo(did, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &max_cu_num, NULL);
  checkError(status, "Query for device info failed");

  status = clGetDeviceInfo(did, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
  checkError(status, "Query for device info failed");

  cl_uint max_work_item_dim;
  status = clGetDeviceInfo(did,  CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS , sizeof(cl_uint), &max_work_item_dim, NULL);
  checkError(status, "Query for device info failed");

  scoped_array<size_t> max_work_item_size(max_work_item_dim);
  status = clGetDeviceInfo(did, CL_DEVICE_MAX_WORK_ITEM_SIZES, max_work_item_dim*sizeof(size_t), max_work_item_size, NULL);
  checkError(status, "Query for device info failed");

  status = clGetDeviceInfo(did, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, NULL);
  checkError(status, "Query for device info failed");

  status = clGetDeviceInfo(did, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, NULL);
  checkError(status, "Query for device info failed");

  status = clGetDeviceInfo(did, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &max_clk_freq, NULL);
  checkError(status, "Query for device info failed");

  printf("Device OpenCL Version: %s\n", version.get());
  printf("Device Max Compute Units: %d\n", (int)max_cu_num);
  printf("Device Max WorkGroup Size: %d\n", (int)max_work_group_size);
  printf("Device Max WorkItem Size: %d\n", (int)max_work_item_size[0]);
  printf("Device Global Memory Size: %d MBytes\n", (int)(global_mem_size/(1024*1024)));
  printf("Device Local Memory Size: %d KBytes\n", (int)local_mem_size/1024);
  printf("Device Max Clock Freq: %d Mhz\n", (int)max_clk_freq);

}


// Create a program for all devices associated with the context.
cl_program createProgramFromFile(cl_context context, const char *kernel_file_name, const cl_device_id *devices, unsigned num_devices) {

  if(!fileExists(kernel_file_name)) {
    printf("Error: Kernel/Binary file '%s' does not exist.\n", kernel_file_name);
    checkError(CL_INVALID_PROGRAM, "Invalid binary file name");
  }

  // Load kernel source or binary from disk
  unsigned char *kernelbinary;
  printf("\nLoading kernel/binary from file %s\n", kernel_file_name);
  size_t binary_size = load_file_to_memory(kernel_file_name, (char **) &kernelbinary);
  if ((int)binary_size < 0) {
    checkError(CL_INVALID_PROGRAM, "Failed to load kernel/binary file");
  }
 
  scoped_array<size_t> binary_lengths(num_devices);
  scoped_array<unsigned char *> binaries(num_devices);
  for(unsigned i = 0; i < num_devices; ++i) {
    binary_lengths[i] = binary_size;
    binaries[i] = kernelbinary; // all device execute the same kernel
  }
  
  cl_int status;
  scoped_array<cl_int> binary_status(num_devices);

#ifdef FPGA_DEVICE
  cl_program program = clCreateProgramWithBinary(context, num_devices, devices, binary_lengths,
      (const unsigned char **) binaries.get(), binary_status, &status);
  checkError(status, "Failed to create program with binary");

  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program with binary");

  for(unsigned i = 0; i < num_devices; ++i) {
    checkError(binary_status[i], "Failed to load binary for device");
  }
#else
  cl_program program = clCreateProgramWithSource(context, 1, (const char **) binaries.get(),
		  0, &status);
  checkError(status, "Failed to create program with source");

  status = clBuildProgram(program, num_devices, devices, NULL, 0, 0);
  checkError(status, "Failed to build program with source");
#endif

  return program;
}


// Randomly generate a floating-point number between -10 and 10.
//#define RAND_MAX 0x7fff
float rand_float()
{
  return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}


//////////////////////////////////////////////////////////////
// Host Memory Allocation Functions for Aligned DMA Operations
//////////////////////////////////////////////////////////////

#ifdef _WIN32 // Windows
void *alignedMalloc(size_t size, size_t alignment) {
	return _aligned_malloc (size, alignment);
}

void alignedFree(void * ptr) {
	_aligned_free(ptr);
}
#else          // Linux
void *alignedMalloc(size_t size, size_t alignment) {
	void *result = NULL;
	posix_memalign (&result, alignment, size);
	return result;
}

void alignedFree(void * ptr) {
	free (ptr);
}
#endif


} //namespace ocl_util
