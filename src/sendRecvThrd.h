// parSMURF
// Alessandro Petrini, 2018-2019
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <thread>
#include <mutex>
#include <mpi.h>
#include "HyperSMURFUtils.h"
#include "MPIHelper.h"
#include "partition.h"
#include "partitionMPI.h"
#include "samplerKNNMPI.h"
#include "globals.h"
#include "rfRanger.h"
#include "ForestProbability.h"
#include "DataDouble.h"
#include "DataFloat.h"

void recvThrd( uint32_t totThrd, uint32_t currentThrd, uint32_t rank, uint32_t partitionForThisProc, uint32_t trngDataSize,
		uint32_t m, uint32_t n, uint32_t fp, uint32_t k, uint32_t numTrees, uint32_t mtry, uint32_t rfThr, uint32_t seed,
	 	uint32_t testSize, double * const testData, double * tempProb1, double * tempProb2, std::mutex * p_accumulLock, bool MPIverbose,
	 	uint8_t wmode, std::string forestDirname );

void sendThrd( uint32_t totThrd, uint32_t currentThrd, uint32_t mm, uint32_t trngDataSize,
		const MPIHelper * const MpiH, std::vector<PartitionMPI> &partMPI, const std::vector<double> & xx, bool MPIverbose );

void predictRecvThrd( uint32_t totThrd, uint32_t currentThrd, uint32_t rank_, uint32_t partitionForThisProc,
	uint32_t m, uint32_t numTrees, uint32_t mtry, uint32_t rfThr, uint32_t seed,
	uint32_t testSize, double * const testData, double * tempProb1, double * tempProb2, std::mutex * p_accumulLock,
	bool MPIverbose, std::string forestDirname );
