/***************************************************************************
Copyright (c) 2019, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, 
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, 
this list of conditions and the following disclaimer in the documentation 
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors 
may be used to endorse or promote products derived from this software 
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

***************************************************************************/
#include "xf_headers.h"
#include "xf_cvt_color_config.h"
#include "xf_resize_config.h"
//#include <CL/cl.h>
#include "xcl2.hpp" 
#include <chrono>

//test !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//test ++++++++++++++++++++++++

using namespace std::chrono;

int main(int argc,char **argv)
{
	if(argc != 2){
		printf("Usage : <executable> <input image> \n");
		return -1;
	}
    char* xclbinFilename = argv[1];

	cv::Mat source_img, source_cvt_img;
	cv::Mat resize_img_ocv, resize_img_hls;
	cv::Mat cvtColor_img_ocv, cvtColor_img_hls;
	cv::Mat error;

	source_img = cv::imread("/home/chaoz/Wen_workspace/k2m2k_1ppc/k2m2k/data/emu_pic.jpg", 1);
	//source_cvt_img = cv::imread("../image/input_frame.jpg", 1);


	if(!source_img.data){
	    std::cout << "No image found. '"  << std::endl;
		return -1;
	}

	int in_width,in_height;
	int out_width,out_height;
	int new_hls_width,new_hls_height;
	int new_cv_width,new_cv_height;

	in_width = source_img.cols;
	in_height = source_img.rows;
	out_height = NEWHEIGHT;
	out_width = NEWWIDTH;

	int depth = source_img.depth();

	resize_img_ocv.create(cv::Size(NEWWIDTH, NEWHEIGHT),source_img.depth());
	resize_img_hls.create(cv::Size(NEWWIDTH, NEWHEIGHT),source_img.depth());

	error.create(cv::Size(NEWWIDTH, NEWHEIGHT),source_img.depth());

	cvtColor_img_ocv.create(cv::Size(source_img.cols, source_img.rows),source_img.depth());
	cvtColor_img_hls.create(source_img.rows, source_img.cols, CV_8UC1);

	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	/*OpenCV cvtColor function*/
	cv::cvtColor(source_img, cvtColor_img_ocv, cv::COLOR_RGB2GRAY);//CV_BGR2GRAY);

	/*OpenCV resize function*/
	#if INTERPOLATION==0
	cv::resize(cvtColor_img_ocv,resize_img_ocv,cv::Size(NEWWIDTH,NEWHEIGHT),0,0,0);
	#endif
	#if INTERPOLATION==1
	cv::resize(cvtColor_img_ocv,resize_img_ocv,cv::Size(NEWWIDTH,NEWHEIGHT),0,0,cv::INTER_LINEAR);
	#endif
	#if INTERPOLATION==2
	cv::resize(cvtColor_img_ocv,resize_img_ocv,cv::Size(NEWWIDTH,NEWHEIGHT),0,0,cv::INTER_AREA);
	#endif

	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);


	cv::cvtColor(source_img, source_img, cv::COLOR_BGR2RGB);//CV_BGR2GRAY);

	/////////////////////////////////////// CL ///////////////////////////////////////
	//BILL START NEW CODE
	   std::vector<cl::Device> devices;
	    cl::Device device;
	    std::vector<cl::Platform> platforms;
	    bool found_device = false;

	    //traversing all Platforms To find Xilinx Platform and targeted
	    //Device in Xilinx Platform
	    cl::Platform::get(&platforms);
	    for(size_t i = 0; (i < platforms.size() ) & (found_device == false) ;i++){
	        cl::Platform platform = platforms[i];
	        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
	        if ( platformName == "Xilinx"){
	            devices.clear();
	            platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
		    if (devices.size()){
			    device = devices[0];
			    found_device = true;
			    break;
		    }
	        }
	    }
	    if (found_device == false){
	       std::cout << "Error: Unable to find Target Device "
	           << device.getInfo<CL_DEVICE_NAME>() << std::endl;
	       return EXIT_FAILURE;
	    }
    // Creating Context and Command Queue for selected device
    cl::Context context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);

    // Load xclbin
    std::cout << "Loading: '" << xclbinFilename << "'\n";
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg (0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg (0, bin_file.beg);
    char *buf = new char [nb];
    bin_file.read(buf, nb);

    // Creating Program from Binary File
    cl::Program::Binaries bins;
    bins.push_back({buf,nb});
    //devices.resize(1);
    //cl::Program program(context, devices, bins);

//BILL END NEW CODE

	devices.resize(1);
	cl::Program program(context, devices, bins);

	cl::Kernel resize_accel(program,"resize_accel");
	cl::Kernel cvtcolor_bgr2gray(program,"cvtcolor_bgr2gray");

	cl::Buffer imageToDevice_cvt(context,CL_MEM_READ_ONLY, in_height*in_width*INPUT_CH_TYPE);
	//cl::Buffer imageFromDevice_cvt(context,CL_MEM_WRITE_ONLY,in_height*in_width*OUTPUT_CH_TYPE);

	cl::Buffer imageK2KinMem(context,CL_MEM_WRITE_ONLY,in_height*in_width*OUTPUT_CH_TYPE);

	//cl::Buffer imageToDevice(context,CL_MEM_READ_ONLY, in_height*in_width);
	cl::Buffer imageFromDevice(context,CL_MEM_WRITE_ONLY,out_height*out_width);

	/* Copy input vectors to memory */
	q.enqueueWriteBuffer(imageToDevice_cvt,CL_TRUE, 0,  in_height*in_width*INPUT_CH_TYPE, (ap_uint<INPUT_PTR_WIDTH>*)source_img.data);
	//q.enqueueWriteBuffer(imageToDevice_cvt,CL_TRUE, 0,  in_height*in_width*INPUT_CH_TYPE, (ap_uint<INPUT_PTR_WIDTH>*)source_img.data);
	
// Set the kernel arguments

	cvtcolor_bgr2gray.setArg(0, imageToDevice_cvt);
	cvtcolor_bgr2gray.setArg(1, imageK2KinMem);
	cvtcolor_bgr2gray.setArg(2, in_height);
	cvtcolor_bgr2gray.setArg(3, in_width);

	resize_accel.setArg(0, imageK2KinMem);
	resize_accel.setArg(1, imageFromDevice);
	resize_accel.setArg(2, in_height);
	resize_accel.setArg(3, in_width);
	resize_accel.setArg(4, out_height);
	resize_accel.setArg(5, out_width);


	// Profiling Objects
	cl_ulong start_cvt= 0;
	cl_ulong end_cvt = 0;
	cl_ulong start_resize= 0;
	cl_ulong end_resize = 0;
	double diff_prof = 0.0f;
	cl::Event event_sp1;
	cl::Event event_sp2;


// Launch the kernel 
	q.enqueueTask(cvtcolor_bgr2gray,NULL,&event_sp1);
	clWaitForEvents(1, (const cl_event*) &event_sp1);
	q.enqueueTask(resize_accel,NULL,&event_sp2);
	clWaitForEvents(1, (const cl_event*) &event_sp2);


	event_sp1.getProfilingInfo(CL_PROFILING_COMMAND_START,&start_cvt);
	event_sp1.getProfilingInfo(CL_PROFILING_COMMAND_END,&end_cvt);
	diff_prof = end_cvt-start_cvt;
	std::cout << "\ncvtColor runtime (fabric): "<<(diff_prof/1000000)<<"ms"<<std::endl;

	event_sp2.getProfilingInfo(CL_PROFILING_COMMAND_START,&start_resize);
	event_sp2.getProfilingInfo(CL_PROFILING_COMMAND_END,&end_resize);
	diff_prof = end_resize-start_resize;
	std::cout << "Resize runtime (fabric): "<<(diff_prof/1000000)<<"ms\n"<<std::endl;

	diff_prof = end_resize-start_cvt;
	std::cout << "Total kernel runtime (fabric): "<<(diff_prof/1000000)<<"ms"<<std::endl;
	std::cout << "OpenCV duration (CPU): " << time_span.count()*1000 << "ms\n" << std::endl;

	//Copying Device result data to Host memory
	q.enqueueReadBuffer(imageFromDevice, CL_TRUE, 0, out_height*out_width, (ap_uint<OUTPUT_PTR_WIDTH>*)resize_img_hls.data);
	//q.enqueueReadBuffer(imageK2KinMem, CL_TRUE, 0, in_height*in_width*OUTPUT_CH_TYPE, (ap_uint<OUTPUT_PTR_WIDTH>*)cvtColor_img_hls.data);

	q.finish();
/////////////////////////////////////// end of CL ///////////////////////////////////////
		


    cv::imwrite("../out_resize_hls.jpg",resize_img_hls);
    cv::imwrite("../out_frame_result_cv.jpg",resize_img_ocv);
    //cv::imwrite("../out_color_hls.jpg",cvtColor_img_hls);
    //cv::imwrite("../out_source_img.jpg",source_img);


    new_hls_width = resize_img_hls.cols;
    new_hls_height = resize_img_hls.rows;
    new_cv_width = resize_img_ocv.cols;
    new_cv_height = resize_img_ocv.rows;
//    std::cout<<"Orig : "<< in_width << " " << in_height << std::endl;
//    std::cout<<"HLS : "<< new_hls_width << " " << new_hls_height << std::endl;
//    std::cout<<"CV : "<< new_cv_width << " " << new_cv_height << std::endl;

	cv::absdiff(resize_img_hls,resize_img_ocv,error);
	float err_per;
	xf::cv::analyzeDiff(error, 10, err_per);

	return 0;
}
