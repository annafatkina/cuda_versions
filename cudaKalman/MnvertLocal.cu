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

//#include "./MnvertLocalWrapper.h"

//__________________________________________________________________________

/*extern "C" 
{
	 void MnvertLocal_cpu(Double_t *a, Int_t l, Int_t n, 
				  Int_t ifail);
}
*/
//__________________________________________________________________________


__device__ void L40_func(int km1, int k, int l, double* a, double* localVERTpp, double* localVERTq)
{
  //  int j = blockIdx.y * blockDim.y + threadIdx.y + 1 + blockIdx.x * blockDim.x + threadIdx.x; // ?????
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (j <= km1) {
//localVERTpp[j-1] = 15.0;
//localVERTq[j-1] = 15.0;
            localVERTpp[j-1] = a[j + k*l];
            localVERTq[j-1]  = a[j + k*l]*localVERTq[k-1];
            a[j + k*l]   = 0;
        }
}

__device__ void L51_func(int kp1, int n, int l, int k, double* a, double* localVERTpp, double* localVERTq)
{
   // int j = blockIdx.y * blockDim.y + threadIdx.y + 1 + blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (j >= kp1 && j <= n) {
            localVERTpp[j-1] = a[k + j*l];
            localVERTq[j-1]  = -a[k + j*l]*localVERTq[k-1];
            a[k + j*l]   = 0;
        }
}


//60
__device__ void ElimProper(int n, int l, double* a, double* localVERTpp, double* localVERTq)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;		// ????!!!!
    int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (j <= n) 
    {
            if(k >= j && k <= n) 
	    { 
//a[j + k*l] = 15.0;
		a[j + k*l] += localVERTpp[j-1]*localVERTq[k-1];  
	    }
    }
}

__device__ void LeftDiadAndUnscaling(int n, int l, double* localVERTs, double* a)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;		// ????!!!!
    int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if(j <= n) 
    {
        if ( k <= j) 
	{
            a[k + j*l] = a[k + j*l]*localVERTs[k-1]*localVERTs[j-1];
            a[j + k*l] = a[k + j*l];
//a[k + j*l] = 15.0;
//a[j + k*l] = 15.0;
        }
    }
}

__device__ void ScaleMatrix_gpu(int n, int l,
				double* a, double* localVERTs)
{//90
    double si;
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;		// ????!!!!
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
   // successScale = &TRUE;						// m b without successScale 
    if(i <= n) {
        si = a[i + i*l];
        if (si <= 0) 
	{
    //        successScale = &FALSE;
	    return;
	}
        localVERTs[i-1] = 1 / sqrt(si);
//localVERTs[i-1]=15.0;
    }
    if(i <= n) {
        if(j <= n) {
//a[i + j*l] = 15.0;
            a[i + j*l] = a[i + j*l]*localVERTs[i-1]*localVERTs[j-1];
        }
    }
}




__global__ void MnvertLocal_gpu(double *a, double* localVERTq, double* localVERTpp, double* localVERTs, 
				int l, int n,  int* ifail)
{
    
    bool* successScale;
    ScaleMatrix_gpu(n, l, a, localVERTs);
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1; //120
    int kp1, km1, k;
/*localVERTs[i] = 15.0;
localVERTpp[i] = 15.0;
localVERTq[i] = 15.0;
*/    if(i<=n) {
        k = i;
//*-*-                  preparation for elimination step1
        if (a[k + k*l] != 0) localVERTq[k-1] = 1 / a[k + k*l];
        else
	{
		*ifail = 1;
		return;
	}
        localVERTpp[k-1] = 1;
        a[k + k*l] = 0;
        kp1 = k + 1;
        km1 = k - 1;
        if (km1 < 0) 
	{
		*ifail = 1;
		return;
	}
        else if (km1 == 0) goto L50;
        else               goto L40;
L40:
        L40_func(km1, k, l, a, localVERTpp, localVERTq);
L50:
        if (k - n < 0) goto L51;
        else if (k - n == 0) goto L60;
        else 	{ *ifail = 1; return; }
L51:
        L51_func(kp1, n, l, k, a, localVERTpp, localVERTq);
//*-*-                  elimination proper		//150
L60:    ElimProper(n, l, a, localVERTpp, localVERTq);

//*-*-                  elements of left diagonal and unscaling
    LeftDiadAndUnscaling(n, l, localVERTs, a);
    return;
//*-*-                  failure return
L100:
    *ifail = 1;
    }
 /* mnvertLocal */
}

extern "C" void MnvertLocal_cpu(Double_t *a, Int_t l, Int_t n, Int_t ifail)
{
  Double_t * localVERTs = new Double_t[n];
  Double_t * localVERTq = new Double_t[n];
  Double_t * localVERTpp = new Double_t[n];

  double * localVERTs_dev;
  double * localVERTq_dev;
  double * localVERTpp_dev;
  double * a_dev;
  cudaMalloc(&a_dev, n*n*sizeof(double));
  cudaMalloc(&localVERTs_dev, n*sizeof(double));
  cudaMalloc(&localVERTq_dev, n*sizeof(double));
  cudaMalloc(&localVERTpp_dev, n*sizeof(double));

  double * localVERTs_host;
  double * localVERTq_host;
  double * localVERTpp_host;
  double * a_host;
  a_host = (double*)malloc(n*n*sizeof(double));
  localVERTs_host = (double*)malloc(n*sizeof(double));
  localVERTq_host = (double*)malloc(n*sizeof(double));
  localVERTpp_host = (double*)malloc(n*sizeof(double));

for (int p = 0; p <n*n; p++)
{
a_host[p] = (double)a[p];
}	
cudaMemcpy(a_dev, a_host, n*n*sizeof(double), cudaMemcpyHostToDevice);
	std::cout  << "*****************************----------------******************"<< std::endl;

	std::cout  << "n = " << n << std::endl << "(int)ceil(n/16 + 0,5) = " << (int)ceil(n/16 + 0.5) << std::endl;

  // fMaxint changed to localMaxint
  Int_t localMaxint = n;

    /* System generated locals */
    Int_t aOffset;

    /* Local variables */
    Double_t si;
    Int_t i, j, k, kp1, km1;

    /* Parameter adjustments */
    aOffset = l + 1;
    a -= aOffset;

    /* Function Body */
    ifail = 0;
    if (n < 1) goto L101;		
    if (n > localMaxint) goto L101;

// here will be gpu call
   // dim3 threadSize = dim3(16, 16);
    //dim3 blockSize = dim3((int)ceil(n/16+0.5), (int)ceil(n/16+0.5));			// count it!!!!!!

   // MnvertLocal_gpu <<<dim3(16, 16), dim3((int)ceil(n/16+0.5), (int)ceil(n/16+0.5))>>> ((double*)a, localVERTq, localVERTpp, localVERTs, 
	//			 (int)l, (int)n, 
	//			 (int*)ifail);
    
/* for (int p = 0; p < n; p++)
    {
	std::cout << localVERTs[p] << " " << localVERTq[p] << " " <<  localVERTpp[p] << std::endl;
    }
	std::cout  << "*****************************----------------******************"<< std::endl;
 */
   MnvertLocal_gpu <<<1, n*n>>> (a_dev, localVERTq_dev, localVERTpp_dev, localVERTs_dev, 
				 (int)l, 
				(int)n, 
				 (int*)ifail);
    
    cudaDeviceSynchronize();
    cudaMemcpy(localVERTs_host, localVERTs_dev, n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(localVERTq_host, localVERTq_dev, n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(localVERTpp_host, localVERTpp_dev, n*sizeof(double), cudaMemcpyDeviceToHost);

    for (int p = 0; p < n; p++)
    {
	//localVERTs[p] = localVERTs_host[p];
	//localVERTq[p] = localVERTq_host[p];
	//localVERTpp[p] = localVERTpp_host[p];
	std::cout << localVERTs_host[p] << " " << localVERTq_host[p] << " " <<  localVERTpp_host[p] << std::endl;
    }
	std::cout  << "*****************************----------------******************"<< std::endl;
//*-*-                  elements of left diagonal and unscaling
    cudaFree(localVERTs_dev);
    cudaFree(localVERTq_dev);
    cudaFree(localVERTpp_dev);

    cudaFree(localVERTs_host);
    cudaFree(localVERTq_host);
    cudaFree(localVERTpp_host);

    delete [] localVERTs;
    delete [] localVERTq;
    delete [] localVERTpp;
    return;
//*-*-                  failure return
L101:
    cudaFree(localVERTs_dev);
    cudaFree(localVERTq_dev);
    cudaFree(localVERTpp_dev);

    cudaFree(localVERTs_host);
    cudaFree(localVERTq_host);
    cudaFree(localVERTpp_host);


    delete [] localVERTs;
    delete [] localVERTq;
    delete [] localVERTpp;
    ifail = 1;
/* mnvertLocal */
}
