#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <math.h>
#include "clean_inputs.h"

#define CL_TARGET_OPENCL_VERSION 300
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define MAX_SOURCE_SIZE (0x100000)
#include <CL/cl.h>

#include <opencv2/opencv.hpp>//For image operations

int print_all_devices(){
  size_t BUFFSIZE = 40;
  size_t BUFF_DEV = 10;

  char cbuffer[BUFFSIZE];


  /*Step1: Getting platforms.*/
  cl_uint numPlatforms;	//the NO. of platforms
  cl_int	status = clGetPlatformIDs(0, NULL, &numPlatforms);
  if (status != CL_SUCCESS)
  {
    std::cout << "Error: Getting platforms!" << std::endl;
    return 71;//EX_OSERR
  }
  if(numPlatforms==0){
    std::cout << "No valid OpenCL platforms found! Exiting..." <<std::endl;
    return 69;//EX_UNAVAILABLE
  }

  //Get Platform IDs
  cl_platform_id* platforms = (cl_platform_id* )malloc(numPlatforms* sizeof(cl_platform_id));
  cl_device_id* devices = (cl_device_id* )malloc(BUFF_DEV* sizeof(cl_device_id));
  status = clGetPlatformIDs(numPlatforms, platforms, NULL);
  cl_uint numDevices = 0;

  std::cout<<"Platforms:"<<std::endl;
  for(int i = 0; i < numPlatforms; i++){
    numDevices = 0;
    status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, BUFF_DEV, devices, &numDevices);
    status = clGetPlatformInfo(platforms[i],CL_PLATFORM_NAME,BUFFSIZE,cbuffer,NULL);
    std::cout<<cbuffer<<" - "<<numDevices<<" device(s)"<<std::endl;
    for(int j = 0; j < numDevices; j++){
      std::cout<<"    ";
      std::cout<<i<<","<<j<<": ";
      status = clGetDeviceInfo(devices[j],CL_DEVICE_NAME,BUFFSIZE,cbuffer,NULL);
      std::cout<<cbuffer<<std::endl;
    }
  }
  return 0;
}

std::vector<float> build_rvec(std::vector<float> range, int n_out){
  float step = (range[1]-range[0])/n_out;
  float cv = range[0];
  std::vector<float> out;
  for(int i = 0; i < n_out; i++){
    out.push_back(cv);
    cv+=step;
  }
  return out;
}

std::vector<uchar> get_cols(std::vector<float> vin, std::vector<float> c1, std::vector<float> c2,int exp_p){
  std::vector<uchar> out;
  for(int i = 0; i < vin.size(); i++){
    //std::vector<uchar> in;
    for(int j = 0; j < c1.size(); j++){
      if(vin[i]<0){
        float fv = exp(vin[i]);
        fv = pow(fv,exp_p);
	fv = fv*c1[j];
        out.push_back((uchar)(int)(255*fv));
      }else{
        float fv = exp(-vin[i]);
        fv = pow(fv,exp_p);
	fv = fv*c2[j];
        out.push_back((uchar)(int)(255*fv));
      }
    }
    //out.push_back(in);
  }
  return out;
}

int render_lyapunov(std::string outname,
                    int WIDTH,
                    int HEIGHT,
                    std::vector<float> ARANG,
                    std::vector<float> BRANG,
                    float x0,
                    int exp_p,
                    int n,
                    std::vector<int> ord,
                    std::vector<float> c1,
                    std::vector<float> c2,
		    int platform_ind,
		    int device_ind)
{
  std::vector<float> v_vals(WIDTH*HEIGHT,x0);
  std::vector<float> d_vals(WIDTH*HEIGHT,0);

  std::vector<float> a_rvs = build_rvec(ARANG,WIDTH);
  std::vector<float> b_rvs = build_rvec(BRANG,HEIGHT);

  int olen = (int)ord.size();

  float* p_v = v_vals.data();
  float* p_d = d_vals.data();
  float* p_a = a_rvs.data();
  float* p_b = b_rvs.data();
  int* p_o = ord.data();

  //OPENCL STUFF
  //Read kernel
  FILE *fp;
  char *source_str;
  size_t source_size;

  fp = fopen("src/logmap_kernel.cl", "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }
  source_str = (char *) malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);

  // Get platform and device information
  cl_platform_id platform_id[platform_ind+1];
  cl_device_id device_ids[device_ind+1];
  cl_device_id device_id;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret = 0;
  ret = clGetPlatformIDs(platform_ind+1, &platform_id[0], &ret_num_platforms);
  if(ret_num_platforms < platform_ind+1){
    fprintf(stderr, "Platform index %i does not exist. Please run with `-l' for OpenCL device info.\n",platform_ind);
    return 1;
  }
  ret = clGetDeviceIDs(platform_id[platform_ind], CL_DEVICE_TYPE_ALL, device_ind+1, &device_ids[0], &ret_num_devices);
  if(ret_num_devices < device_ind+1){
    fprintf(stderr, "Device index %i does not exist for platform %i. Please run with `-l' for OpenCL device info.\n",device_ind,platform_ind);
    return 1;
  }
  device_id = device_ids[device_ind];


  //Context
  cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

  // Create a command queue
  cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

  // Create memory buffers on the device for each vector
  cl_mem cl_v = clCreateBuffer(context, CL_MEM_READ_WRITE, v_vals.size() * sizeof(float), NULL, &ret);
  cl_mem cl_d = clCreateBuffer(context, CL_MEM_READ_WRITE, d_vals.size() * sizeof(float), NULL, &ret);
  cl_mem cl_a = clCreateBuffer(context, CL_MEM_READ_ONLY, a_rvs.size() * sizeof(float), NULL, &ret);
  cl_mem cl_b = clCreateBuffer(context, CL_MEM_READ_ONLY, b_rvs.size() * sizeof(float), NULL, &ret);
  cl_mem cl_o = clCreateBuffer(context, CL_MEM_READ_ONLY, ord.size() * sizeof(int), NULL, &ret);

  // Copy the data to buffers
  ret = clEnqueueWriteBuffer(command_queue, cl_v, CL_TRUE, 0, v_vals.size() * sizeof(float), p_v, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, cl_d, CL_TRUE, 0, d_vals.size() * sizeof(float), p_d, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, cl_a, CL_TRUE, 0, a_rvs.size() * sizeof(float), p_a, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, cl_b, CL_TRUE, 0, b_rvs.size() * sizeof(float), p_b, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, cl_o, CL_TRUE, 0, ord.size() * sizeof(int), p_o, 0, NULL, NULL);

  // Create a program from the kernel source
  cl_program program = clCreateProgramWithSource(context, 1, (const char **) &source_str, (const size_t *) &source_size, &ret);

  // Build the program
  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  if (ret==-11) {
    // Determine the size of the log
    size_t log_size;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

    // Allocate memory for the log
    char *log = (char *) malloc(log_size);

    // Get the log
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

    // Print the log
    printf("%s\n", log);
  }


  // Create the OpenCL kernel
  cl_kernel kernel = clCreateKernel(program, "logmap", &ret);

  // Set the arguments of the kernel
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &cl_v);
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &cl_a);
  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &cl_b);
  ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &cl_o);
  ret = clSetKernelArg(kernel, 4, sizeof(int), (void *) &olen);
  ret = clSetKernelArg(kernel, 5, sizeof(int), (void *) &n);
  ret = clSetKernelArg(kernel, 6, sizeof(int), (void *) &WIDTH);
  ret = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *) &cl_d);

  int lgs_targ = 64;
  while(((WIDTH*HEIGHT)%lgs_targ)!=0){
    lgs_targ = (int)lgs_targ/2;
  }
  int lg1_targ = lgs_targ;
  while((HEIGHT%lg1_targ)!=0){
    lg1_targ = (int)lg1_targ/2;
  }
  int lg0_targ = (int)lgs_targ/lg1_targ;


  // Execute the OpenCL kernel on the list
  size_t global_item_size[2] = {(size_t)WIDTH,(size_t)HEIGHT}; // Process the entire lists
  size_t local_item_size[2] = {(size_t)lg0_targ,(size_t)lg1_targ}; // Divide work items into groups of 64
  cl_event event;
  cl_event event2;

  ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, &global_item_size[0], &local_item_size[0], 0, NULL, &event);
  clWaitForEvents(1, &event);
  ret = clEnqueueReadBuffer(command_queue, cl_d, CL_TRUE, 0, d_vals.size() * sizeof(float), &p_d[0], 1, &event, NULL);

  //Clean Memory
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(cl_v);
  ret = clReleaseMemObject(cl_d);
  ret = clReleaseMemObject(cl_a);
  ret = clReleaseMemObject(cl_b);
  ret = clReleaseMemObject(cl_o);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);

  auto colim = get_cols(d_vals, c1, c2,exp_p);

  cv::Mat img_o(HEIGHT,WIDTH,CV_8UC3,colim.data());
  cv::cvtColor(img_o, img_o, cv::COLOR_BGR2RGB);
  cv::imwrite(outname, img_o);

  cv::imshow("Display window", img_o);
  cv::waitKey(0);

  return 0;
}

int main(int argc, char* argv[]){
  bool lflag = 0;//Print all devices when used
  int dflag = 0;//Used specific device id

  int index;
  int c;

  std::string outname = "out.png";
  int WIDTH  = 512;
  int HEIGHT = 256;
  std::vector<float> ARANG = {3,4};
  std::vector<float> BRANG = {3,4};
  float x0 = 0.5;
  int exp_p = 4;
  int n = 100000;
  std::vector<int> ord = {0,1};
  std::vector<float> c1 = {0,0,1};
  std::vector<float> c2 = {1,0,0};

  std::string c1r = "";
  std::string c2r = "";

  std::vector<int> cl_dev = {0,0};

  opterr = 0;

  //Parse Inputs
  while ((c = getopt (argc, argv, "lw:h:a:b:x:s:o:p:n:1:2:c:")) != -1)
    switch (c)
      {
      case 'l':
        lflag = 1;
        break;
      case 'n':
        n = clean_int(optarg);
        if(n==-1){
          fprintf (stderr, "Bad argument given to `-%c'.\n", 'n');
          return -1;
        }
        break;
      case 'w':
        WIDTH = clean_int(optarg);
        if(WIDTH==-1){
          fprintf (stderr, "Bad argument given to `-%c'.\n", 'w');
          return -1;
        }
        break;
      case 'h':
        HEIGHT = clean_int(optarg);
        if(HEIGHT==-1){
          fprintf (stderr, "Bad argument given to `-%c'.\n", 'h');
          return -1;
        }
        break;
      case 'a':
        ARANG = clean_floata(optarg);
        if(ARANG.size()!=2){
          fprintf (stderr, "Bad argument given to `-%c'.\n", 'a');
          return -1;
        }
        break;
      case 'b':
        BRANG = clean_floata(optarg);
        if(BRANG.size()!=2){
          fprintf (stderr, "Bad argument given to `-%c'.\n", 'b');
          return -1;
        }
        break;
      case 'x':
        x0 = clean_float(optarg);
        if(x0==-1){
          fprintf (stderr, "Bad argument given to `-%c'.\n", 'x');//COULD BE -1 actually
          return -1;
        }
        break;
      case 's':
        ord = clean_inta(optarg);
        if(ord.size()==0){
          fprintf (stderr, "Bad argument given to `-%c'.\n", 's');
          return -1;
        }
        break;
      case 'o':
        outname = optarg;
        break;
      case 'p':
        exp_p = clean_int(optarg);
        if(exp_p==-1){
          fprintf (stderr, "Bad argument given to `-%c'.\n", 'p');
          return -1;
        }
        break;
      case '1':
        c1r = clean_col(optarg);
        if (c1r==""){
          fprintf (stderr, "Bad argument given to `-%c'.\n", '1');
          return -1;
        }
        break;
      case '2':
        c2r = clean_col(optarg);
        if (c2r==""){
          fprintf (stderr, "Bad argument given to `-%c'.\n", '2');
          return -1;
        }
        break;

      case 'c':
	cl_dev = clean_inta(optarg);
	if(cl_dev.size()==0){
	  fprintf(stderr,"Bad argument given to `-%c'.\n",'c');
	}
	break;
      case '?':
        if(isprint (optopt))
          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
        else
          fprintf (stderr,
                   "Unknown option character `\\x%x'.\n",
                   optopt);
        return 1;
      default:
        abort ();
      }

  if(lflag){
    int ret = print_all_devices();
    return ret;
  }

  if(c1r != ""){
    c1 = hex2col(c1r);
  }
  if(c2r != ""){
    c2 = hex2col(c2r);
  }

  int profile_ind = cl_dev[0];
  int device_ind = cl_dev[1];

  int ret = render_lyapunov(outname,WIDTH,HEIGHT,ARANG,BRANG,x0,exp_p,n,ord,c1,c2,profile_ind,device_ind);
	return ret;
}
