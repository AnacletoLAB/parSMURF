// parSMURF
// Alessandro Petrini, 2018-2019
#include "folds.h"

Folds::Folds( uint32_t numberOfFolds, uint32_t classSize, const uint32_t * labels, std::vector<uint32_t>& ff, bool randomFold ) :
		classSize( classSize ), nFolds( numberOfFolds ), labels( labels ),
		fromFile( ff ), totPos( 0 ), totNeg( 0 ), randomFold( randomFold ) {

	nPos = 		new uint32_t[nFolds];
	nNeg = 		new uint32_t[nFolds];
	foldsIdx = 	new uint32_t[nFolds + 1];
	folds =		new uint32_t[classSize];

}

Folds::~Folds() {
	tempNegIdx.clear();
	tempPosIdx.clear();
	delete[] folds;
	delete[] foldsIdx;
	delete[] nNeg;
	delete[] nPos;
}

// Positive == minor class ||||| Negative == major class
void Folds::computeFolds() {
	if (randomFold) {
		// temp vect clear
		tempPosIdx.clear();
		tempNegIdx.clear();

		// count of positive labels ( == 1) of labels array
		totPos = (uint32_t) std::count_if( labels, labels + classSize, []( int nn ){return nn >= 1; } );
		totNeg = classSize - totPos;

		int pos = totPos, neg = totNeg;
		for (uint32_t i = 0; i < classSize; i++)
			(labels[i] > 0) ? tempPosIdx.push_back( i ) : tempNegIdx.push_back( i );

		// Shuffling... maybe not really mandatory here...
		std::random_shuffle( tempPosIdx.begin(), tempPosIdx.end() );
		std::random_shuffle( tempNegIdx.begin(), tempNegIdx.end() );

		int ffSize = classSize / nFolds;
		int ffRemn = classSize % nFolds;

		// Calculating each fold size
		// If classSize is not a multiple of nFolds, lowest ranking folds will be bigger
		// ex: 100 classes, 6 folds => fold sizes: 17 17 17 17 16 16
		foldsIdx[0] = 0;
		for (uint32_t i = 0; i < nFolds; i++) {
			foldsIdx[i + 1] = foldsIdx[i] + ffSize;
			if (ffRemn != 0) {
				foldsIdx[i + 1]++;
				ffRemn--;
			}
		}

		// clear nPos and nNeg
		std::fill( nPos, nPos + nFolds, 0 );
		std::fill( nNeg, nNeg + nFolds, 0 );

		// Filling the folds, starting with the positives
		int index = 0;
		for (int i = 0; i < pos; i++) {
			index = foldsIdx[i % nFolds] + i / nFolds;
			folds[index] = tempPosIdx[i];
			nPos[i % nFolds]++;
		}
		// Then taking care of the negatives
		for (int i = pos; i < pos + neg; i++) {
			index = foldsIdx[i % nFolds] + i / nFolds;
			folds[index] = tempNegIdx[i - pos];
			nNeg[i % nFolds]++;
		}
		// Do not reorder: first elements of each fold are positives.

		tempPosIdx.clear();
		tempNegIdx.clear();

	} else {
		// Reading the folds from file
		// Total complexity is 3 * nn... can we do better?

		// count of positive labels ( == 1) of labels array
		totPos = (uint32_t) std::count_if( labels, labels + classSize, []( int nn ){return nn >= 1; } );
		totNeg = classSize - totPos;

		// clear nPos and nNeg
		std::fill( nPos, nPos + nFolds, 0 );
		std::fill( nNeg, nNeg + nFolds, 0 );

		// creating an histogram of the values of the fold file
		std::vector<uint32_t> hist( nFolds, 0 );
		std::for_each( fromFile.begin(), fromFile.end(), [&hist]( uint32_t nn ) mutable { hist[nn]++; } );

		// accumulate the histogram as a prefix sum into the foldsIdx array
		foldsIdx[0] = 0;
		for (uint32_t i = 1; i < nFolds + 1; i++) {
			foldsIdx[i] = foldsIdx[i - 1] + hist[i - 1];
		}

		// Now that I know the start and end index of each fold, I can populate it
		std::fill( hist.begin(), hist.end(), 0 );
		for (uint32_t i = 0; i < classSize; i++) {
			uint32_t bin = fromFile[i];
			uint32_t addr = foldsIdx[bin] + hist[bin];
			hist[bin]++;
			if (labels[i] > 0) nPos[bin]++; else nNeg[bin]++;
			folds[addr] = i;
		}
	}

	// Find the max of the nPos and nNeg arrays, since this values will be used for the allocation
	// of the training and test matrices
	maxPos = *(std::max_element( nPos, nPos + nFolds ));
	maxNeg = *(std::max_element( nNeg, nNeg + nFolds ));
}

void Folds::verbose() {
	std::cout << " *** Folds generation ***" << std::endl;
	std::cout << "totPos: " << totPos << " - totNeg: " << totNeg << std::endl << std::endl;
	for (uint32_t i = 0; i < nFolds; i++) {
		std::cout << "Fold: " << i << " - Fold size: " << foldsIdx[i + 1] - foldsIdx[i] << std::endl;
		std::cout << "Fold begins at: " << foldsIdx[i] << " - Fold ends at: " << foldsIdx[i + 1] - 1 << std::endl;
		std::cout << "nPos: " << nPos[i] << " - nNeg: " << nNeg[i] << std::endl;
		/*std::cout << "Positive: ";
		uint32_t foldBegins = foldsIdx[i];
		uint32_t foldSz = foldsIdx[i + 1] - foldsIdx[i];
		for (uint32_t k = 0; k < nPos[i]; k++)
			std::cout << folds[foldBegins + k] << " ";
		std::cout << std::endl;
		std::cout << "Negative: ";
		for (uint32_t k = nPos[i]; k < foldSz; k++)
			std::cout << folds[foldBegins + k] << " ";
		std::cout << std::endl;*/
		std::cout << std::endl;
	}

}

void Folds::setLabelsPtr( const uint32_t * labels ) {
	this->labels = labels;
}
