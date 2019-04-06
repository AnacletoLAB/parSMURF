// parSMURF
// Alessandro Petrini, 2018-2019
#include <complex>
#include "samplerKNN.h"
#include "ANN.h"


void SamplerKNN::sample() {
	setup();
	minOversample();
	majUndersample();
	trngSize = trngPos + trngNeg;
}

void SamplerKNN::setup() {
	// Calculates in advance the total number of positive and negative samples that
	// the training set will be constituted of. We just need the number of
	// positive and negative samples in the current partition.
	posToBeGenerated = (part->fp + 1) * posForOverSmpl;
	negToBeGenerated = posToBeGenerated * part->ratio;
	// If ratio == 0 => disable the undersampler
	if (negToBeGenerated == 0)
		negToBeGenerated = negForUndrSmpl;
	//if (negToBeGenerated > negForUndrSmpl) {
	//	std::cout << "WARNING - fold: " << currentFold << " part.: " << currentPartition << " insuff. negatives: " << negToBeGenerated <<
	//	" requested but " << negForUndrSmpl << " available." << std::endl;
	//}
	negToBeGenerated = (negToBeGenerated > negForUndrSmpl) ? negForUndrSmpl : negToBeGenerated;
	lineLen = posToBeGenerated + negToBeGenerated;
}

void SamplerKNN::minOversample() {
	// Saving the size of the positive testset
	trngPos = posToBeGenerated;

	// Check that there's enough room in the training matrix. There should be, unless maxSize formula
	// is wrong...
	if (trngPos > part->maxSize) {
		std::cerr << "error: training matrix is too small... check maxSize formula (that shouldn't have happened, anyway...)" << std::endl;
		std::cerr << "[in SamplerKNN::minOversample()]" << std::endl;
		abort();
	}

	if (trngPos < 2) {
		std::cerr << "error: we need at least two positive samples in each training partition for oversampling to happen" << std::endl;
		std::cerr << "[in SamplerKNN::minOversample()]" << std::endl;
		abort();
	}

	// k is supposed to be >= of the number of positive examples, but on smaller
	// sets this might not happen (put a "like" on Valgrind Facebook page!)
	uint32_t localk = (posForOverSmpl >= k) ? k : posForOverSmpl;

	// Copy the original positives in the training matrix
	for (uint32_t i = 0; i < posForOverSmpl; i++) {
		copySample( pos[i], i, lineLen );		// copy on Part->trngData
	}

	// Generazione dei rimanenti con KNN
	ANNpointArray		dataPts = annAllocPts( posForOverSmpl, m + 1 );	// Data points. Enough: we do the search only on the positive, not on the entire example set!
	ANNpoint			queryPt = annAllocPt( m + 1 );		// query point
	ANNidxArray			nnIdx	= new ANNidx[localk];		// near neighbor indices
	ANNdistArray		dists	= new ANNdist[localk];		// near neighbor distances
	std::vector<double>	temp(m + 1);

	// load data into dataPts
	for (uint32_t i = 0;  i < posForOverSmpl; i++) {
		getSample( pos[i], dataPts[i] );
	}

	ANNkd_tree kdTree(dataPts, posForOverSmpl, m + 1);			// search structure

	// A note about paragraph 2.2.3 on Ann programming guide: it is stated that ANN_ALLOW_SELF_MATCH is true
	// by default, therefore the nearest neighbour returned by annkSearch (the point nnIdx[0]) should always
	// be the query point itself. We should discuss with Giorgio if this point should be included in the generated
	// point set - I guess no, and in this case it is sufficient to assign:
	// idx = (rand() % k) + 1
	// to ignore idx == 0, and modify the initial check on k as:
	// k = (k < part->posForOverSmpl) ? k + 1 : part->posForOverSmpl - 1;

	// This will be the index for accessing the trngData array via the setSample function;
	// therefore, it should range from part->posForOverSmpl to (posToBeGenerated - 1)
	// (I'm going to assert that at the end of the cicle...)
	uint32_t ptIndex = posForOverSmpl;
	// Pre-declare scalars alpha and idx
	uint32_t idx;
	double alpha;
	// More optimization: cast once (we could move randMax declaration in class declaration and its assignment in the constructor)
	double randMax = static_cast<double>( RAND_MAX );
	// Even more optimizations: minimize the getSample and annkSearh calls.
	for (uint32_t i = 0; i < posForOverSmpl; i++) {
		getSample( pos[i], queryPt );
		kdTree.annkSearch( queryPt, localk, nnIdx, dists, 0 );
		for (uint32_t j = 0; j < part->fp; j++) {
			idx = rand() % (localk - 1);
			//alpha = rand() / randMax;					// randMax is already a double
			getSample( pos[nnIdx[idx]], temp.data() );	// GG, is it right now?
			for (uint32_t l = 0; l < m; l++) {			// to m and not to m+1, because...
				alpha = rand() / randMax;
				temp[l] = temp[l] * alpha + queryPt[l] * (1 - alpha);
			}
			temp[m] = 1.0;								// ...the last item of the array is the label: it should not be interpolated!
			setSample( ptIndex, temp.data(), lineLen );
			ptIndex++;
		}
	}
	// The assert we were talking about...
	if (ptIndex != posToBeGenerated) {
		std::cerr << "error: mismatch in point generation:" << std::endl;
		std::cerr << "  ptIndex = " << ptIndex << " --- posToBeGenerated = " << posToBeGenerated << std::endl;
		std::cerr << "[in SamplerKNN::minOversample()]" << std::endl;
		abort();
	}

	annDeallocPts( dataPts );
	annDeallocPt( queryPt );
	annClose();
	delete[] nnIdx;
	delete[] dists;
}

void SamplerKNN::majUndersample() {

	// Size of the positive test set (original + generated)
	uint32_t generatedPos = trngPos;

	// Guess what?
	if (trngPos + negToBeGenerated > part->maxSize) {
		std::cerr << "error: training matrix is too small... check maxSize formula (that shouldn't have happened, anyway...)" << std::endl;
		std::cerr << "[in SamplerKNN::majUndersample()]" << std::endl;
		abort();
	}

	// copy the points
	for (uint32_t i = 0; i < negToBeGenerated; i++) {
		copySample( neg[i], i + generatedPos, lineLen );
	}

	// and save the size of the negative set
	trngNeg = negToBeGenerated;


}

void SamplerKNN::verbose() {
	double * dataOut = trngData;

	std::cout << "  *** KNN Sampler *** " << std::endl;
	std::cout << "Data size: " << trngSize << " - Positives: " << trngPos << " - Negatives: " << trngNeg << std::endl;
	std::cout << "Original positives" << std::endl;
	for (uint32_t i = 0; i < posForOverSmpl; i++) {
		std::cout << std::setw( 4 ) << i << " | " << std::setw( 4 ) << pos[i] << " | ";
		for (uint32_t k = 0; k < m + 1; k++) {
			std::cout << std::setw( 10 ) << dataOut[i + lineLen * k] << " | ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	std::cout << "Generated positives" << std::endl;
	for (uint32_t i = posForOverSmpl; i < trngPos; i++) {
		std::cout << std::setw( 4 ) << i << " | " << std::setw( 4 ) << "xxx" << " | ";
		for (uint32_t k = 0; k < m + 1; k++) {
			std::cout << std::setw( 10 ) << dataOut[i + lineLen * k] << " | ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	dataOut = trngData + trngPos;
	std::cout << "Undersampled negatives" << std::endl;
	for (uint32_t i = 0; i < trngNeg; i++) {
		std::cout << std::setw( 4 ) << i + trngPos << " | " << std::setw( 4 ) << neg[i] << " | ";
		for (uint32_t k = 0; k < m + 1; k++) {
			std::cout << std::setw( 10 ) << dataOut[i + lineLen * k] << " | ";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;
	std::cout << "Raw part->trngData view:" << std::endl;
	for (uint32_t i = 0; i < trngSize * (m + 1); i++) {
		std::cout << trngData[i] << " ";
		if ((i + 1) % (lineLen) == 0) std::cout << std::endl;
	}
	std::cout << std::endl;
}
