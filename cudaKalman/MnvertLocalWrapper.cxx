#include <TMath.h>
#include <TGeoManager.h>
#include <TClonesArray.h>
#include <TLorentzVector.h>
#include <cuda.h>
#include "MnvertLocalWrapper.h"
extern void MnvertLocal_cpu(Double_t *a, Int_t l, Int_t n, 
				  Int_t ifail);

MnvertLocalWrapper::MnvertLocalWrapper()
{
}

MnvertLocalWrapper::~MnvertLocalWrapper()
{
}

void MnvertLocalWrapper::MnvertLocalCpu(Double_t *a, Int_t l, Int_t n, 
				  Int_t ifail) 
{
	MnvertLocal_cpu((double*) a, (int)l,  (int)n, 
				  (int)ifail);
}

ClassImp(MnvertLocalWrapper)
