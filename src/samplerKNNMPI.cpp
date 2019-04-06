// parSMURF
// Alessandro Petrini, 2018-2019
#include "samplerKNNMPI.h"

void SamplerKNNMPI::sample() {
	// majUndersample has already been performed by master MPI proc
	minOversample();
}

void SamplerKNNMPI::minOversample() {
	// k is supposed to be >= of the number of positive examples, but on smaller
	// sets this might not happen (put a "like" on Valgrind Facebook page!)
	uint32_t localk = (posForOverSmpl >= k) ? k : posForOverSmpl;

	// Generazione dei rimanenti con KNN
	ANNpointArray		dataPts = annAllocPts( posForOverSmpl, m + 1 );	// Data points. Enough: we do the search only on the positive, not on the entire example set!
	ANNpoint			queryPt = annAllocPt( m + 1 );		// query point
	ANNidxArray			nnIdx	= new ANNidx[localk];		// near neighbor indices
	ANNdistArray		dists	= new ANNdist[localk];		// near neighbor distances
	double*				temp	= new double[m + 1];

	// load data into dataPts
	for (uint32_t i = 0;  i < posForOverSmpl; i++) {
		getSample( i, dataPts[i] );
	}

	ANNkd_tree*			kdTree = nullptr;					// search structure
	kdTree = new ANNkd_tree(dataPts, posForOverSmpl, m + 1);    // Saving memory, as before

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
		getSample( i, queryPt );
		kdTree->annkSearch( queryPt, localk, nnIdx, dists, 0 );
		for (uint32_t j = 0; j < fp; j++) {
			idx = rand() % (localk - 1);
			//alpha = rand() / randMax;					// randMax is already a double
			getSample( nnIdx[idx], temp );
			for (uint32_t l = 0; l < m; l++) {			// to m and not to m+1, because...
				alpha = rand() / randMax;
				temp[l] = temp[l] * alpha + queryPt[l] * (1 - alpha);
			}
			temp[m] = 1.0;								// ...the last item of the array is the label: it should not be interpolated!
			setSample( ptIndex, temp );
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

	delete kdTree;
	annDeallocPts( dataPts );
	annDeallocPt( queryPt );
	annClose();
	delete[] temp;
	delete[] nnIdx;
	delete[] dists;

}

void SamplerKNNMPI::verbose() {
}

void SamplerKNNMPI::accumulateTempProb( const std::vector<std::vector<std::vector<double>>>& predictions, const uint32_t testSize,
		double * const class1, double * const class2 ) {

	for (uint32_t i = 0; i < testSize; i++) {
		class1[i] += predictions[0][i][0];
		class2[i] += predictions[0][i][1];
	}
}


void SamplerKNNMPI::getSample( const uint32_t numSamp, double * const sample ) {
	for (uint32_t i = 0; i < m + 1; i++)
		sample[i] = x[numSamp + lineLen * i];
}

void SamplerKNNMPI::setSample( const uint32_t numCol, const double * const sample ) {
	for (uint32_t i = 0; i < m + 1; i++)
		x[numCol + lineLen * i] = sample[i];
}
