// parSMURF
// Alessandro Petrini, 2018-2019
#include "curves.h"

// TODO: fix AUROC inconsistent results (on linux only)

Curves::Curves() : preds( nullptr ), labls( std::vector<uint32_t>(0) ) {}

Curves::Curves( const std::vector<uint32_t> & labels, const double * const predictions ) :
		labls( labels ), preds( predictions ) {

	precision	= std::vector<double>( labls.size(), 0 );	// Dummy init
	recall		= std::vector<double>( labls.size(), 0 );	// Dummy init
	recall2		= std::vector<float>( labls.size() + 1 );
	fpr			= std::vector<float>( labls.size() + 1 );
	TP			= std::vector<uint32_t>( labls.size() );
	FP			= std::vector<uint32_t>( labls.size() );
	totP = (uint32_t) std::count( labls.begin(), labls.end(), 1 );
	totN = (uint32_t) std::count( labls.begin(), labls.end(), 0 );

	uint32_t idx = 0;
	tempLabs = std::vector<uint8_t>( labls.size() );
	tempPreds = std::vector<float>( labls.size() );
	// -- create a copy of the labs array for fast arithmetics
	std::for_each( tempLabs.begin(), tempLabs.end(), [&](uint8_t &val) { val = (uint8_t) labls[idx++];} );
	// -- create a copy of the predictions array for fast arithmetics
	idx = 0;
	std::for_each( tempPreds.begin(), tempPreds.end(), [&](float &val) { val = (float) preds[idx++];} );

	// - Sort the labels by descending prediction scores
	idx = 0;
	indexes = std::vector<size_t>( labls.size() );
	// -- sort the indexes based on prediction values
	std::generate( indexes.begin(), indexes.end(), [&idx]() {return idx++;} );
	std::sort( indexes.begin(), indexes.end(), [&](size_t i, size_t j) { return preds[i] > preds[j];} );
	// -- apply the indexes permutation to labs and preds vectors
	apply_permutation_in_place<uint8_t, float>( tempLabs, tempPreds, indexes );
}

Curves::~Curves() {}

void Curves::evalAUPRCandAUROC( double * const out ) {
	out[1] = evalAUROC_ok();
	out[0] = evalAUPRC();
}

double Curves::evalAUROC_alt() {
	uint32_t tempTP = 0;
	uint32_t tempFP = 0;
	float prevPred = -100.0;

	// printVectC<uint8_t>( tempLabs );
	// printVect<float>( tempPreds );

	// - in a loop
	uint32_t idx = 0;
	uint32_t partialIdx = 0;
	while ((idx < tempLabs.size()) /*&& (partialIdx < tempLabs.size() - 1)*/) {
		if (tempPreds[idx] != prevPred) {
			recall2[idx] = tempTP / (float) totP;
			fpr[idx] = tempFP / (float) totN;
			prevPred = tempPreds[idx];
			partialIdx++;
		}
		if (tempLabs[idx] == 1)
			tempTP++;
		else
			tempFP++;
		idx++;
	}

	recall2[partialIdx] = tempTP / (float) totP;
	fpr[partialIdx] = tempFP / (float) totN;
	recall2.resize( partialIdx + 1 );
	fpr.resize( partialIdx + 1 );

	std::cout << "PartialIdx: " << partialIdx << std::endl;

	// - calculate AUROC area by trapezoidal integration
	return traps_integrate<float>( fpr, recall2 );
	// - return AUROC area
}

double Curves::evalAUPRC() {
	// - filter predictions and lables associated with distinct score values
	//filter_dups<float, uint8_t>( tempPreds, tempLabs );
	// - thresholds are the distinct scores
	alphas.clear();
	alphas = tempPreds;

	TP = cumulSum<uint32_t, uint8_t>( tempLabs );
	std::for_each( tempLabs.begin(), tempLabs.end(), [&](uint8_t &val) { val = 1 - val;} );
	FP = cumulSum<uint32_t, uint8_t>( tempLabs );
	FP[FP.size() - 1] = totN;

	// uint32_t idxTP = 0;
	// for (size_t i = 0; i < TP.size(); i++) {
	// 	if (TP[idxTP] == TP[TP.size() - 1])
	// 		break;
	// 	idxTP++;
	// }

	// Calculate precision and recall
	precision.clear();
	recall.clear();
	recall.push_back( 0.0 );
	precision.push_back( 1.0 );
	for (size_t i = 0; i < FP.size(); i++) {
		precision.push_back( TP[i] / (double) (TP[i] + FP[i]) );
		recall.push_back( TP[i] / (double) totP );
	}
	//
	// recall.push_back( 1.0 );
	// precision.push_back( 0.0 );

	// recall.resize( idxTP + 2 );
	// precision.resize( idxTP + 2 );

	return traps_integrate<double>( recall, precision );
}


double Curves::evalAUROC_ok() {
	std::vector<uint8_t> tempLabs2 = tempLabs;
	std::vector<float>  tempPreds2 = tempPreds;
//	filter_dups<float, uint8_t>( tempPreds2, tempLabs2 );
	alphas.clear();
	alphas = tempPreds2;

	TP = cumulSum<uint32_t, uint8_t>( tempLabs2 );
	std::for_each( tempLabs2.begin(), tempLabs2.end(), [&](uint8_t &val) { val = 1 - val;} );
	FP = cumulSum<uint32_t, uint8_t>( tempLabs2 );
	FP[FP.size() - 1] = totN;

	// uint32_t idxTP = 0;
	// for (size_t i = 0; i < TP.size(); i++) {
	// 	if (TP[idxTP] == TP[TP.size() - 1])
	// 		break;
	// 	idxTP++;
	// }

	fpr.clear();
	recall2.clear();
	// recall2.push_back( 0.0 );
	// fpr.push_back( 0.0 );
	for (size_t i = 0; i < FP.size(); i++) {
		fpr.push_back( FP[i] / (float) totN );
		recall2.push_back( TP[i] / (float) totP );
	}

	// fpr.push_back( 1.0f );
	// recall2.push_back( 0.0 );

	return traps_integrate<float>( fpr, recall2 );
}
