// parSMURF
// Alessandro Petrini, 2018-2019
#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <mpi.h>

#include "HyperSMURFUtils.h"

#define TRNGHEADERSIZE 5

// Map for trngData:
// trngData[0]: current fold
// trngData[1]: current partition
// trngData[2]: training set size
// trgnData[3]: number of positive on training set (posForOverSmpl)
// trgnData[4]: number of positive on training set AFTER oversampling (posToBeGenerated)
// trngData[TRNGHEADERSIZE]: beginning of real data

class MPIHelper {
public:
	MPIHelper( uint32_t worldsize, uint32_t nFolds, uint32_t ensThrd, uint32_t rfThrd, bool distributedFolds, bool MPIverbose );
	~MPIHelper();
	void verbose();

	static void saveToFileMPI( const double * const cl1, const double * const cl2, const uint32_t nn,
			const std::vector<uint32_t> * const labels, std::string outFilename );

	void broadcastGlobVars( int rank, uint32_t * const nn, uint32_t * const mm, uint32_t * const nFolds );
	void broadcastMtrys( int rank, std::vector<GridParams> & gridParams );
	void updateChunks( uint32_t newParts );

	uint32_t				worldSize;
	uint32_t				numberOfMasterProcs;
	uint32_t				numberOfWorkingProcs;
	uint32_t				numberOfThreadsPerProc;

	uint32_t				ensThrd;
	uint32_t				rfThrd;

	uint32_t				nParts;
	uint32_t				numberOfChunks;
	std::vector<uint32_t>	numberOfPartsPerChunk;
	std::vector<uint32_t>	startingPartition;
	bool					distributedFolds;
	bool					MPIverbose;
};
