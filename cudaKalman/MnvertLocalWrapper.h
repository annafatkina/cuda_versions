#ifndef MNVERTWRAPPER_H_
#define MNVERTWRAPPER_H_

#include <TMath.h>
#include <TGeoManager.h>
#include <TClonesArray.h>
#include <TLorentzVector.h>


class MnvertLocalWrapper:public TObject
{
public:
	MnvertLocalWrapper();
	virtual ~MnvertLocalWrapper();

	void MnvertLocalCpu(Double_t *a, Int_t l, Int_t n, 
				  Int_t ifail);	
	ClassDef(MnvertLocalWrapper,1);

};


#endif /* MNVERTWRAPPER_H_ */