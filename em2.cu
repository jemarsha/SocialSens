#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <stdio.h>
#include <math.h>
#include <cstring>

using namespace std;


__global__ void compute_z(int *NOC_device,int *NOS_device,int *SC_device,float *a_device,float *b_device,float *Z_device,float *d_device){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < NOC_device[0]){
        float A = 0;
	float B = 0;
	for (int i = 0; i < NOS_device[0]; i++) {
  	    A = A + SC_device[i*NOC_device[0]+id] * log(a_device[i]) + (1-SC_device[i*NOC_device[0]+id]) * log(1 - a_device[i]);
	    B = B + SC_device[i*NOC_device[0]+id] * log(b_device[i]) + (1-SC_device[i*NOC_device[0]+id]) * log(1 - b_device[i]);
	    //printf("A[%d] is: %.10f  a[i]=%f B[%d] is: %.10f  b[i]=%f \n",i,log(a_device[i]),a_device[i],i,log(b_device[i]),b_device[i]);
        }
	A = exp(A);
	B = exp(B);
	Z_device[id] = (A * d_device[0]) / ((A * d_device[0]) + (B * (1 - d_device[0])));
	//printf("Z[%d] is: %.10f \n",id,Z_device[id]);
	//printf("A[%d] is: %.10f \n",id,log(a_device[i]);
	//printf("B[%d] is: %.10f \n",id,log(b_device[i]);	
	//printf("d[%d] is: %.10f \n",id,d_device[0]);
    } 
}


__global__ void compute_theta(int *NOC_device,int *NOS_device,int *SC_device,float *a_device,float *b_device,float *Z_device,float *d_device){
        int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id < NOS_device[0]){
	    float tempz = 0;
	    float totalz = 0;
	    int cnt = 0;
	    for (int j = 0; j < NOC_device[0]; j++) {
	        if (SC_device[id * NOC_device[0] +j] == 1) {
		    tempz = tempz + Z_device[j];
		    cnt = cnt + 1;
		}
  		totalz = totalz + Z_device[j];
	    }
	    a_device[id] = tempz / totalz;
	    b_device[id] = (cnt - tempz) / (NOC_device[0] - totalz);
	    d_device[0] = totalz / NOC_device[0];
	}    
}


int main() {


    FILE *input = fopen("TestSensingMatrix.txt", "r");

    const int NOS = 30;
    const int NOC = 2000;
    const int MAX_IT = 10;

    int *NOS_device;
    int *NOC_device;
    int *MAX_IT_device;
    cudaMalloc(&NOS_device, sizeof(int));
    cudaMalloc(&NOC_device, sizeof(int));
    cudaMalloc(&MAX_IT_device, sizeof(int));
    cudaMemcpy( NOS_device, &NOS, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( NOC_device, &NOC, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( MAX_IT_device, &MAX_IT, sizeof(int), cudaMemcpyHostToDevice);

    cudaError_t malloc_error_check = cudaGetLastError();
    if(malloc_error_check != cudaSuccess){
    
        printf("malloc_error_check: CUDA error: %s\n", cudaGetErrorString(malloc_error_check));
        exit(-1);
    }

    int SC[NOS*NOC];
    std::memset(SC, 0, sizeof(SC));

    int *SC_device;
    cudaMalloc(&SC_device, NOS*NOC*sizeof(int));
    cudaMemset(SC_device, 0, sizeof(int)*NOS*NOC);

    // Generate the SC matrix
    int row[2];
    while (fscanf(input, "%d,%d", &row[0], &row[1]) == 2) {
        SC[(row[0] - 1)*NOC + (row[1] - 1)] = 1;
    }
    
    cudaMemcpy(SC_device, SC, sizeof(int)*NOS*NOC, cudaMemcpyHostToDevice); 
    
    cudaError_t malloc_error_check2 = cudaGetLastError();
    if(malloc_error_check2 != cudaSuccess){
    
        printf("malloc_error_check2: CUDA error: %s\n", cudaGetErrorString(malloc_error_check2));
        exit(-1);
    }

    // s
    float s[NOS];

    float *s_device;
    cudaMalloc(&s_device, NOS*sizeof(float));

    std::memset(s, 0, sizeof(s));
    for (int x = 0; x < NOS; x++) {
        int cnt = 0;
        for (int y = 0; y < NOC; y++) {
            if (SC[x*NOC+y] == 1) {
                cnt = cnt + 1;
            }
        }
        s[x] = cnt * 1.0 / NOC;
    }

    cudaMemcpy(s_device, s, sizeof(float)*NOS, cudaMemcpyHostToDevice);

    // theta[ai]
    float a[NOS];
    float b[NOS];
    float d = 0.5;
    float *d_device;
    cudaMalloc(&d_device, sizeof(float));
    cudaMemcpy(d_device, &d, sizeof(float), cudaMemcpyHostToDevice);

    float Z[NOC];

    std::memset(a, 0, sizeof(a));
    std::memset(b, 0, sizeof(b));
    std::memset(Z, 0, sizeof(Z));


    float *a_device;
    float *b_device;
    float *Z_device;
    cudaMalloc(&a_device, NOS*sizeof(float));
    cudaMalloc(&b_device, NOS*sizeof(float));
    cudaMalloc(&Z_device, NOC*sizeof(float));
    cudaMemset(Z_device, 0, sizeof(float)*NOC);
    cudaMemset(a_device, 0, sizeof(float)*NOS);
    cudaMemset(b_device, 0, sizeof(float)*NOS);

    cudaError_t malloc_error_check3 = cudaGetLastError();
    if(malloc_error_check3 != cudaSuccess){
    
        printf("malloc_error_check3: CUDA error: %s\n", cudaGetErrorString(malloc_error_check3));
        exit(-1);
    }

    for (int x = 0; x < NOS; x++) {
        a[x] = s[x];
        b[x] = 0.5 * s[x];
    }

    cudaMemcpy(a_device, a, sizeof(float)*NOS, cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b, sizeof(float)*NOS, cudaMemcpyHostToDevice);
 

    cudaError_t malloc_error_check4 = cudaGetLastError();
    if(malloc_error_check4 != cudaSuccess){

        printf("malloc_error_check4: CUDA error: %s\n", cudaGetErrorString(malloc_error_check4));
        exit(-1);
    }
 
    dim3 grid_vertex((int)ceil((float)NOC/(float)1024),1), block_vertex(1024,1);
    dim3 grid_vertex2((int)ceil((float)NOS/(float)1024),1), block_vertex2(1024,1); 


    for (int itn = 0; itn < MAX_IT; itn++) {
        // Compute Z(t, j)
        compute_z<<<grid_vertex,block_vertex>>>(NOC_device,NOS_device,SC_device,a_device,b_device,Z_device,d_device);
        //for(int foo1=0;foo1<NOC;foo1++){
          //  printf("z[%d] = %f \n",foo1,z2[foo1]);
        //}
        cudaError_t iter_error = cudaGetLastError();
        if(iter_error != cudaSuccess)
        {
            printf("iter_error: CUDA error: %s\n", cudaGetErrorString(iter_error));
            exit(-1);
        }
	cudaDeviceSynchronize();
        compute_theta<<<grid_vertex2,block_vertex2>>>(NOC_device,NOS_device,SC_device,a_device,b_device,Z_device,d_device);
	cudaDeviceSynchronize();
    }

    cudaError_t vertex_filter_errorri = cudaGetLastError();
    if(vertex_filter_errorri != cudaSuccess)
    {
        printf("FilterFrontierrri: CUDA error: %s\n", cudaGetErrorString(vertex_filter_errorri));
        exit(-1);
    }

    // end of while
    compute_z<<<grid_vertex,block_vertex>>>(NOC_device,NOS_device,SC_device,a_device,b_device,Z_device,d_device);

    cudaError_t vertex_filter_errorr2 = cudaGetLastError();
    if(vertex_filter_errorr2 != cudaSuccess)
    {
        printf("FilterFrontierrr2: CUDA error: %s\n", cudaGetErrorString(vertex_filter_errorr2));
        exit(-1);
    }


    cudaMemcpy(Z, Z_device, sizeof(float)*NOC, cudaMemcpyDeviceToHost);
    cudaMemcpy(a, a_device, sizeof(float)*NOS, cudaMemcpyDeviceToHost);
    cudaMemcpy(b, b_device, sizeof(float)*NOS, cudaMemcpyDeviceToHost);
    cudaMemcpy(&d, d_device, sizeof(float), cudaMemcpyDeviceToHost);

    FILE *groundtruth = fopen("TestGroundTruth.txt", "r");
    int gt[NOC];
    std::memset(gt, 0, sizeof(gt));

    while (fscanf(input, "%d,%d", &row[0], &row[1]) == 2) {
        gt[row[0] - 1] = row[1];
    }

    int out[NOC];
    std::memset(out, 0, sizeof(out));
   
    FILE *output = fopen("outtie.txt", "w");
    for (int j = 0; j < NOC; j++) {
    	if (Z[j] >= 0.5) {
	    	out[j] = 1;
        }
    }
    float t[NOS];
    std::memset(t, 0, sizeof(t));

    for (int i = 0; i < NOS; i++) {
      t[i] = (a[i]*d) / ((a[i]*d) + (b[i]*(1-d)));
        cout << t[i] << endl;
    }

    
    for (int j = 0; j < NOC; j++) {
	    fprintf(output, "%d, %d\n", j+1, out[j]);
    }

    fclose(input); 
 
    //Free GPU Memory
    cudaFree(NOS_device);
    cudaFree(NOC_device);
    cudaFree(MAX_IT_device);
    cudaFree(SC_device);
    cudaFree(s_device);
    cudaFree(d_device);
    cudaFree(a_device); 
    cudaFree(b_device);
    cudaFree(Z_device);  
}

