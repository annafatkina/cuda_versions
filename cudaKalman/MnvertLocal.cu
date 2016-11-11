/// 
/// Kalman filter track reconstructor in the MPD detector - MnvertLocal function cuda version
/// \author Anna Fatkina

#include "MpdKalmanFilter.h"
#include "MpdKalmanTrack.h"
#include "MpdKalmanHit.h"
#include "MpdKalmanGeoScheme.h"
#include "MpdCodeTimer.h"
#include "FairField.h"
//#include "FairRootManager.h"
#include "FairRunAna.h"
#include "FairTask.h"
#include "MpdConstField.h"
#include "MpdMultiField.h"
#include <TMath.h>
#include <TGeoManager.h>
#include <TClonesArray.h>
#include <TLorentzVector.h>

#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <malloc.h>

#define NPP_MINABS_64F ( 1e-322 )


__device__ void L40Func (double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail, int kp1, int km1, int k);

__device__ void L50Func (double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail, int kp1, int km1, int k);

__device__ void L51Func(double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail, int kp1, int km1, int k);

__device__ void L60Func(double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail);

__device__ void ElenLeft(double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail);

__device__ void MainLoop(double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail, int* localFail);

__device__ void ScaleMatrix(double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail, int *localFail);

__device__ void AfterScaleMatrix(double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail, int* localFail);

__global__ void MnvertLocal_gpu (double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail, int* localFail);



// ****************
__device__ void L40Func (double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail, int kp1, int km1, int k)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
 //int j =  threadIdx.x + 1;
  if (j <= km1) 
  {
    localVERTpp_dev[j-1] = a_dev[j + k*l];

    localVERTq_dev[j-1]  = __dmul_rn(a_dev[j + k*l], localVERTq_dev[k-1]);
    a_dev[j + k*l]   = 0.0;
  }
  L50Func(a_dev, localVERTs_dev, localVERTq_dev, localVERTpp_dev, n, l, ifail, kp1, km1, k);

}

__device__ void L50Func (double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail, int kp1, int km1, int k)
{
  if (k - n < 0) 
  {
    L51Func(a_dev, localVERTs_dev, localVERTq_dev, localVERTpp_dev, n, l, ifail, kp1, km1, k);
  }
  else if (k - n == 0) 
  {
    L60Func(a_dev, localVERTs_dev, localVERTq_dev, localVERTpp_dev, n, l, ifail);
  }
  else 
  {
    ifail[0] = 50;   //?????? never

  }
}
__device__ void L51Func(double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail, int kp1, int km1,int k)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
 // int j =  threadIdx.x + 1;
 
  if (j >= kp1 && j <= n) 
  {
    localVERTpp_dev[j-1] = a_dev[k + j*l];
    
    localVERTq_dev[j-1]  = __dmul_rn(-a_dev[k + j*l],localVERTq_dev[k-1]);
    a_dev[k + j*l]   = 0.0;

  }
  L60Func(a_dev, localVERTs_dev, localVERTq_dev, localVERTpp_dev, n, l, ifail);
}
__device__ void L60Func(double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
 // int j = threadIdx.x + 1;
 // int k = threadIdx.y + 1;
    if (j <= n) 
    {
      if (k >= j && k <= n) 
      { 
        a_dev[j + k*l] = __dadd_rn(a_dev[j + k*l], __dmul_rn(localVERTpp_dev[j-1],localVERTq_dev[k-1]));
        
      }
    }
  //  ElenLeft(a_dev, localVERTs_dev, localVERTq_dev, localVERTpp_dev, n, l, ifail);
}

__device__ void ElenLeft(double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
  //int j = threadIdx.x + 1;
  //int k = threadIdx.y + 1;
      if(j <= n) {
        if (k <= j) {
            a_dev[k + j*l] = __dmul_rn(__dmul_rn(a_dev[k + j*l],localVERTs_dev[k-1]),localVERTs_dev[j-1]);
            a_dev[j + k*l] = a_dev[k + j*l];
            
        }
    }
}

__device__ void MainLoop(double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail, int* localFail)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
//  int i = threadIdx.x + 1;
  
  int kp1, km1, k;
  if (i <= n) {
    k = i;
  //*-*-                  preparation for elimination step1
    if (fabs(a_dev[k + k*l]) < NPP_MINABS_64F) 
      {
        *localFail = 1;
        ifail[i-1] = 1;
        return;
        
      }
    else 
    {
      localVERTq_dev[k-1] = __drcp_rn(a_dev[k + k*l]);

    }
    localVERTpp_dev[k-1] = 1.0;
    a_dev[k + k*l] = 0.0;
    kp1 = k + 1;
    km1 = k - 1;
    if (km1 < 0) 
      {
        ifail[i-1] = 2;
        *localFail = 1;
        return;
      }
      else if (km1 == 0) 
      {
        L50Func(a_dev, localVERTs_dev, localVERTq_dev, localVERTpp_dev, n, l, ifail,  kp1, km1, k);
      }
      else
      {
        L40Func(a_dev, localVERTs_dev, localVERTq_dev, localVERTpp_dev, n, l, ifail, kp1, km1, k);
      }
    }
}

__device__ void AfterScaleMatrix(double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail, int* localFail)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  //int i = threadIdx.x + 1;
  //int j = threadIdx.y + 1;
  if (i <= n) 
  {
    if (j <= n)
    {
      a_dev[i + j*l] = __dmul_rn(__dmul_rn(a_dev[i + j*l], localVERTs_dev[i-1]), localVERTs_dev[j-1]);

    }
  }
}
__device__ void ScaleMatrix(double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail, int* localFail)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  //int i = threadIdx.x + 1;
  //int j = threadIdx.y + 1;
  
  double si;
  if (i <= n)
  {
    si = a_dev[i + i*l];
    if (si <= 0) 
    {
      ifail[i-1] = 3;
      *localFail=1;
      return;
    }
    localVERTs_dev[i-1] =__drcp_rn(__dsqrt_rn(si));  }
  AfterScaleMatrix( a_dev,  localVERTs_dev, localVERTq_dev, localVERTpp_dev, n, l,ifail, localFail);
  
}

__global__ void MnvertLocal_gpu (double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail, int* localFail)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
//int j =threadIdx.x + 1;
 // int k = threadIdx.y + 1;
/*if (j <= n)
  {
    localVERTs_dev[j-1] = 33;
  }
  */
  if (j <= n)
  {
    localVERTs_dev[j-1] = 0.0;
    localVERTq_dev[j-1] = 0.0;
    localVERTpp_dev[j-1] = 0.0;
    
    ifail[j-1] = 0;
  }
 *localFail = 0;
  ScaleMatrix(a_dev, localVERTs_dev, localVERTq_dev, localVERTpp_dev, n, l, ifail, localFail);
  
  if(*localFail == 1)
  {
    return;
  }
  *localFail = 0;
  MainLoop(a_dev, localVERTs_dev, localVERTq_dev, localVERTpp_dev, n, l, ifail, localFail);
  if(*localFail == 1)
  {
    return;
  }
  /*if (j <= n) 
  {
    if (k <= j) 
    {
      a_dev[k + j*l] = a_dev[k + j*l]*localVERTs_dev[k-1]*localVERTs_dev[j-1];
      a_dev[j + k*l] = a_dev[k + j*l];
    }
  }*/
  ElenLeft(a_dev, localVERTs_dev, localVERTq_dev, localVERTpp_dev, n, l, ifail);

}

//__________________________________________________________________________
extern "C" void MnvertLocal_cpu(Double_t *a, Int_t l, Int_t n, 
          int* ifail)
{
std::cout <<"!!!!!!HRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR!!!!";
  cudaSetDevice(0);
  // taken from TMinuit package of Root (l>=n)
  // fVERTs, fVERTq and fVERTpp changed to localVERTs, localVERTq and localVERTpp
  //  double_t localVERTs[n], localVERTq[n], localVERTpp[n];
/*  double_t * localVERTs = new double_t[n];
  double_t * localVERTq = new double_t[n];
  double_t * localVERTpp = new double_t[n];
  std::cout << "default: " << localVERTs[3] << std::endl;
*/
//***************************************************//
//std::cout << "size a = " << _msize(a)/a[0];
  double * localVERTs_dev;
  double * localVERTq_dev;
  double * localVERTpp_dev;
  double * a_dev;
  int * ifail_dev;
  cudaMalloc((void**)&ifail_dev, n*sizeof(int));
  cudaMalloc((void**)&a_dev, n*n*sizeof(double));
  cudaMalloc((void**)&localVERTs_dev, n*sizeof(double));
  cudaMalloc((void**)&localVERTq_dev, n*sizeof(double));
  cudaMalloc((void**)&localVERTpp_dev, n*sizeof(double));

  double * localVERTs_host;
  double * localVERTq_host;
  double * localVERTpp_host;
  double * a_host;
  int * ifail_host;
  ifail_host = (int*)malloc(n*sizeof(int));
  a_host = (double*)malloc(n*n*sizeof(double));
  localVERTs_host = (double*)malloc(n*sizeof(double));
  localVERTq_host = (double*)malloc(n*sizeof(double));
  localVERTpp_host = (double*)malloc(n*sizeof(double));

cudaDeviceSynchronize();

//**************************************************//


   /* int k=1;
    int *i;
    i=&k;
    std::cout<<*i;
*/
  // fMaxint changed to localMaxint
  Int_t localMaxint = n;

    /* System generated locals */
    Int_t aOffset;

    /* Local variables */
    //double_t si;
    //Int_t kp1, km1;

    aOffset = l + 1;
    a -= aOffset;
    
  for (int p = 0; p <n*n; ++p)
  {
    a_host[p] = (double)(a[p]) ;
  } 
  cudaMemcpy(a_dev, a_host, n*n*sizeof(double), cudaMemcpyHostToDevice);
  std::cout << "after memcopy to host from host, a:" << std::endl;
  for (int p = 0; p <n*n; ++p)
  {
    std::cout << a_host[p] << " ";
  } 

   int * localFail;
   cudaMalloc((void**)&localFail, sizeof(int));
  //*localFail = 0;

  std::cout << std::endl;
cudaDeviceSynchronize();
    /* Function Body */
   *ifail = 0;
   std::cout << "ifail making 0" << std::endl;
    if (n < 1)       goto L100;
    if (n > localMaxint) goto L100;
  std::cout << "Before cuda func " << std::endl;
    MnvertLocal_gpu<<< n, n>>>(a_dev, localVERTs_dev, localVERTq_dev, localVERTpp_dev, (int)n, (int)l, ifail_dev, localFail);
cudaDeviceSynchronize();
    cudaMemcpy(localVERTs_host, localVERTs_dev, n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(localVERTq_host, localVERTq_dev, n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(localVERTpp_host, localVERTpp_dev, n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(a_host, a_dev, n*n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ifail_host, ifail_dev, n*sizeof(int), cudaMemcpyDeviceToHost);
cudaDeviceSynchronize(); 
std::cout << "after memcopy, ifail:" << std::endl;
for(int p = 0; p < n; ++p)
{
  std::cout << ifail_host[p] << " ";
}
//ifail = *ifail_host;
std::cout << "after ifail, a:" << std::endl;
  for (int p = 0; p <n*n; ++p)
  {
    a[p] = (Double_t)(a_host[p]) ;
    std::cout << a[p] << " ";
  } 
std::cout << "after changing a" << std::endl;
/*  for (int p = 0; p < n; p++)
  {
    localVERTs[p] =  (double_t)localVERTs_host[p];
    localVERTq[p] =  (double_t)localVERTq_host[p];
    localVERTpp[p] = (double_t)localVERTpp_host[p];
  }
  */std::cout << "after changing 3 arrays" << std::endl;
/* for (int p = 0; p < n; p++)
    {
  std::cout << localVERTs[p] << " " << localVERTq[p] << " " <<  localVERTpp[p] << std::endl;
    }
  std::cout  << "*****************************----------------******************"<< std::endl;
for(int p = 0; p < n*n; p++)
{
std::cout << a[p] << " ";
}*/
 /* for (int p = 0; p <n*n; ++p)
  {
    a[p] = (double_t)(a_host[p]) ;
  }*/ 

for(int p = 0; p < n; p++)
{
  std::cout << ifail_host[p] << " ";
  if (ifail_host[p] != 0)
  {

    *ifail = 1;
    //break;
  }
}
std::cout<< "after ifail count" << std::endl;
for (int p = 0; p < n; p++)
    {
  std::cout << localVERTs_host[p] << " " << localVERTq_host[p] << " " <<  localVERTpp_host[p] << std::endl;
    }
  std::cout  << "*****************************----------------******************"<< std::endl;

    cudaFree(localVERTs_dev);
    cudaFree(localVERTq_dev);
    cudaFree(localVERTpp_dev);
    cudaFree(a_dev);
    cudaFree(ifail_dev);
    cudaFree(localFail);



    free(localVERTs_host);
    free(localVERTq_host);
    free(localVERTpp_host);
    free(a_host);
    free(ifail_host);
    
   /* delete [] localVERTs;
    delete [] localVERTq;
    delete [] localVERTpp;
    */return;
//*-*-                  failure return
L100:

std::cout << "IF IFAIL:" << std::endl;
for (int p = 0; p < n; p++)
    {
  std::cout << localVERTs_host[p] << " " << localVERTq_host[p] << " " <<  localVERTpp_host[p] << std::endl;
    }
  std::cout  << "*****************************----------------******************"<< std::endl;

 /*for (int p = 0; p < n; p++)
    {
  std::cout << localVERTs[p] << " " << localVERTq[p] << " " <<  localVERTpp[p] << std::endl;
    }
  std::cout  << "*****************************----------------******************"<< std::endl;

for(int p = 0; p < n*n; p++)
{
std::cout << a[p] << " ";
}*/
    cudaFree(localVERTs_dev);
    cudaFree(localVERTq_dev);
    cudaFree(localVERTpp_dev);
    cudaFree(a_dev);
    cudaFree(ifail_dev);
    cudaFree(localFail);
    

    free(localVERTs_host);
    free(localVERTq_host);
    free(localVERTpp_host);
    free(a_host);
    free(ifail_host);
  /*  delete [] localVERTs;
    delete [] localVERTq;
    delete [] localVERTpp;
    */
    *ifail = 1;
} /* mnvertLocal */
