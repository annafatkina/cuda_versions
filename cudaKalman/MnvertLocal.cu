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

__device__ void L40Func (double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail, int kp1, int km1, int k);

__device__ void L50Func (double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail, int kp1, int km1, int k);

__device__ void L51Func(double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail, int kp1, int km1, int k);

__device__ void L60Func(double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail);

__device__ void ElenLeft(double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail);

__device__ void MainLoop(double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail);

__device__ void ScaleMatrix(double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail);

__global__ void MnvertLocal_gpu (double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail);



// ****************
__device__ void L40Func (double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail, int kp1, int km1, int k)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (j <= km1) 
  {
    localVERTpp_dev[j-1] = a_dev[j + k*l];
    localVERTq_dev[j-1]  = a_dev[j + k*l]*localVERTq_dev[k-1];
    a_dev[j + k*l]   = 0;
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
    *ifail = 1;
  }
}
__device__ void L51Func(double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail, int kp1, int km1,int k)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (j >= kp1 && j <= n) 
  {
    localVERTpp_dev[j-1] = a_dev[k + j*l];
    localVERTq_dev[j-1]  = -a_dev[k + j*l]*localVERTq_dev[k-1];
    a_dev[k + j*l]   = 0;
  }
  L60Func(a_dev, localVERTs_dev, localVERTq_dev, localVERTpp_dev, n, l, ifail);
}
__device__ void L60Func(double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (j <= n) 
    {
      if (k >= j && k <= n) 
      { 
        a_dev[j + k*l] += localVERTpp_dev[j-1]*localVERTq_dev[k-1]; 
      }
    }
}

__device__ void ElenLeft(double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
      if(j <= n) {
        if (k <= j) {
            a_dev[k + j*l] = a_dev[k + j*l]*localVERTs_dev[k-1]*localVERTs_dev[j-1];
            a_dev[j + k*l] = a_dev[k + j*l];
        }
    }
}

__device__ void MainLoop(double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int kp1, km1, k;
  if (i <= n) {
    k = i;
  //*-*-                  preparation for elimination step1
    if (a_dev[k + k*l] != 0) 
      {
        localVERTq_dev[k-1] = 1 / a_dev[k + k*l];
      }
    else 
    {
      *ifail = 1;
      return;
    }
    localVERTpp_dev[k-1] = 1;
    a_dev[k + k*l] = 0;
    kp1 = k + 1;
    km1 = k - 1;
    if (km1 < 0) 
      {
        *ifail = 1;
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

__device__ void ScaleMatrix(double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  double si;
  if (i <= n)
  {
    si = a_dev[i + i*l];
    if (si <= 0) 
    {
      *ifail = 1;
      return;
    }
    localVERTs_dev[i-1] = 1 / sqrt(si);
  }
  if (i <= n) 
  {
    if (j <= n)
    {
      a_dev[i + j*l] = a_dev[i + j*l]*localVERTs_dev[i-1]*localVERTs_dev[j-1];
    }
  }
}

__global__ void MnvertLocal_gpu (double* a_dev, double* localVERTs_dev, double* localVERTq_dev, double* localVERTpp_dev, int n, int l, int *ifail)
{

  ScaleMatrix(a_dev, localVERTs_dev, localVERTq_dev, localVERTpp_dev, n, l, ifail);
  MainLoop(a_dev, localVERTs_dev, localVERTq_dev, localVERTpp_dev, n, l, ifail);
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (j <= n) 
  {
    if (k <= j) 
    {
      a_dev[k + j*l] = a_dev[k + j*l]*localVERTs_dev[k-1]*localVERTs_dev[j-1];
      a_dev[j + k*l] = a_dev[k + j*l];
    }
  }
}

//__________________________________________________________________________
extern "C" void MnvertLocal_cpu(Double_t *a, Int_t l, Int_t n, 
          Int_t &ifail)
{
  cudaSetDevice(0);
  // taken from TMinuit package of Root (l>=n)
  // fVERTs, fVERTq and fVERTpp changed to localVERTs, localVERTq and localVERTpp
  //  Double_t localVERTs[n], localVERTq[n], localVERTpp[n];
  Double_t * localVERTs = new Double_t[n];
  Double_t * localVERTq = new Double_t[n];
  Double_t * localVERTpp = new Double_t[n];


//***************************************************//

  double * localVERTs_dev;
  double * localVERTq_dev;
  double * localVERTpp_dev;
  double * a_dev;
  int * ifail_dev;
  cudaMalloc(&ifail_dev, sizeof(int));
  cudaMalloc(&a_dev, n*n*sizeof(double));
  cudaMalloc(&localVERTs_dev, n*sizeof(double));
  cudaMalloc(&localVERTq_dev, n*sizeof(double));
  cudaMalloc(&localVERTpp_dev, n*sizeof(double));

  double * localVERTs_host;
  double * localVERTq_host;
  double * localVERTpp_host;
  double * a_host;
  int * ifail_host;
  ifail_host = (int*)malloc(sizeof(int));
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
    //Double_t si;
    //Int_t kp1, km1;

    aOffset = l + 1;
    a -= aOffset;
    
  for (int p = 0; p <n*n; ++p)
  {
    a_host[p] = (double)(a[p]) ;
  } 
    cudaMemcpy(a_dev, a_host, n*n*sizeof(double), cudaMemcpyHostToDevice);
cudaDeviceSynchronize();
    /* Function Body */
   //ifail = 0;
    if (n < 1)       goto L100;
    if (n > localMaxint) goto L100;
    MnvertLocal_gpu<<< dim3(n),dim3(n, n)>>>(a_dev, localVERTs_dev, localVERTq_dev, localVERTpp_dev, n, l, ifail_dev);
cudaDeviceSynchronize();
    cudaMemcpy(localVERTs_host, localVERTs_dev, n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(localVERTq_host, localVERTq_dev, n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(localVERTpp_host, localVERTpp_dev, n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(a_host, a_dev, n*n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ifail_host, ifail_dev, sizeof(double), cudaMemcpyDeviceToHost);
cudaDeviceSynchronize(); 
ifail = *ifail_host;
/* for (int p = 0; p < n; p++)
    {
  std::cout << localVERTs[p] << " " << localVERTq[p] << " " <<  localVERTpp[p] << std::endl;
    }
  std::cout  << "*****************************----------------******************"<< std::endl;
for(int p = 0; p < n*n; p++)
{
std::cout << a[p] << " ";
}*/
  for (int p = 0; p <n*n; ++p)
  {
    a[p] = (Double_t)(a_host[p]) ;
  } 
    cudaFree(localVERTs_dev);
    cudaFree(localVERTq_dev);
    cudaFree(localVERTpp_dev);
    cudaFree(a_dev);


    free(localVERTs_host);
    free(localVERTq_host);
    free(localVERTpp_host);
    free(a_host);
    delete [] localVERTs;
    delete [] localVERTq;
    delete [] localVERTpp;
    return;
//*-*-                  failure return
L100:


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


    free(localVERTs_host);
    free(localVERTq_host);
    free(localVERTpp_host);
    free(a_host);
    delete [] localVERTs;
    delete [] localVERTq;
    delete [] localVERTpp;
    ifail = 1;
} /* mnvertLocal */
