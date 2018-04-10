#include <cuda.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>


#define N 64 

#define BLOCK_SIZE 1024

void FncAverage(float *,float *,int);//Function prototype declaration
 

__global__ void parallel_reduction(float* Md, float* Rd) { 

unsigned int tx = threadIdx.x; 
unsigned int bx = blockIdx.x;


__shared__ float partial_max[BLOCK_SIZE];  


partial_max[tx] = Md[bx*blockDim.x+tx];

__syncthreads();

for (unsigned int stride=blockDim.x >>1;stride>0;stride>>=1){
__syncthreads();

if (tx<stride)
partial_max[tx]=max(partial_max[tx], partial_max[tx+stride]);}

if (tx==0){
Rd[bx+tx]=partial_max[tx];}
}





int main()

{
    int n=N;
    int m=N;
    
    int size=n*m;
    
    int sizeR=(size/BLOCK_SIZE);

    float *M_h = (float*) malloc( sizeof(float)*size ); //Memory Allocation in CPU memory

    for (int i=0; i < size; i++)

   { M_h[i] = (rand()%500); } //Creating Array
   
   
   float *R_h = (float*) malloc( sizeof(float)*sizeR );
   
        
    
    float P_ch=0; float R_ch=0.00; 

printf("\n");
printf("Maximum_Value(CPU):");
printf("\n");

for (int i=0; i < size; i++)

   { P_ch = max(P_ch,M_h[i]); } 

printf("%6.2f",P_ch);


FncAverage( M_h, R_h, n );//Calling the vecadd function

for (int i=0; i<sizeR; i++)
{ R_ch = max(R_ch,R_h[i]); }

printf("\n");
printf("Maximum_Value(GPU):");
printf("\n");

printf("%6.2f",R_ch);
printf("\n");



// free the memory we allocated on the CPU

    free( M_h);free(R_h);

   

    return 0;

}



void FncAverage(float *M_h, float *R_h,int n) 
{

int mn=n*n;
int size = mn*sizeof(float);

int sizeR=(mn/BLOCK_SIZE)*sizeof(float);

float *Md; float *Rd; 

    cudaMalloc((void **) &Md, size);//device memory allocation
    cudaMemcpy(Md, M_h, size, cudaMemcpyHostToDevice);// Transfer M_h to device memory 
    
    cudaMalloc((void **) &Rd, sizeR);//device memory allocation
   
    
     
// Initialize the Block and Thread dimension


//Launch Kernel
parallel_reduction<<<ceil(mn/1024), 1024>>> (Md,Rd);


// Transfer Rd from device to host
     cudaMemcpy(R_h, Rd, sizeR, cudaMemcpyDeviceToHost);
       // Free device memory for Md, Rd
     cudaFree(Md); cudaFree(Rd);

}

