// parSMURF
// Alessandro Petrini, 2018-2019
#include "MPIHelper.h"

MPIHelper::MPIHelper( uint32_t worldsz, uint32_t nFolds, uint32_t ensT, uint32_t rfT, bool distributedFolds, bool MPIverbose ) :
		worldSize( worldsz ), ensThrd( ensT ), rfThrd( rfT ), distributedFolds( distributedFolds ), MPIverbose( MPIverbose ) {

	if (!distributedFolds) {
		numberOfMasterProcs = 1;
	} else {
		numberOfMasterProcs = nFolds;
	}
	numberOfWorkingProcs = worldSize - 1;

	// check...
	if (numberOfWorkingProcs <= 0) {
		std::cout << TXT_BIRED << "Not enough working processes! (exec as mpirun -np XX)" << TXT_NORML << std::endl;
		MPI_Finalize();
		exit(-1);
	}

	numberOfThreadsPerProc = ensThrd;
	numberOfChunks = numberOfWorkingProcs;
	numberOfPartsPerChunk = std::vector<uint32_t>( numberOfChunks );
	startingPartition = std::vector<uint32_t>( numberOfChunks + 1 );

}

void MPIHelper::updateChunks( uint32_t newParts ) {
	nParts = newParts;
	uint32_t i = 0;
	std::for_each( numberOfPartsPerChunk.begin(), numberOfPartsPerChunk.end(), [&](uint32_t &nn)
		{ nn = (nParts / numberOfChunks) + ((nParts % numberOfChunks) > i++); } );
	startingPartition[0] = 0;
	for (uint32_t i = 1; i < numberOfChunks + 1; i++)
		startingPartition[i] = ( startingPartition[i - 1] + numberOfPartsPerChunk[i - 1] );
}

void MPIHelper::broadcastGlobVars( int rank, uint32_t * const nn, uint32_t * const mm, uint32_t * const nFolds ) {
	uint32_t globVars[3];
	if (rank == 0) {
		globVars[0] = *nn;
		globVars[1] = *mm;
		globVars[2] = *nFolds;
	}
	if ((rank == 0) & (MPIverbose)) {
		std::cout << TXT_BIPRP << "Broadcasting global variables: nn = " << *nn << " - mm = " << *mm << " - nFolds = " << *nFolds << TXT_NORML << std::endl;
	}
	MPI_Bcast( globVars, 3, MPI_UNSIGNED, 0, MPI_COMM_WORLD );
	MPI_Barrier( MPI_COMM_WORLD );
	if (rank > 0) {
		*nn = globVars[0];
		*mm = globVars[1];
		*nFolds = globVars[2];
	}
	if (MPIverbose) {
		std::cout << TXT_BIPRP << "Rank " << rank << " received from broadcast: nn = " << *nn << " - mm = " << *mm << " - nFolds = " << *nFolds << TXT_NORML << std::endl;
	}
	if ((rank == 0) & (MPIverbose)) {
		std::cout << TXT_BIPRP << "Broadcasting global variables complete." << TXT_NORML << std::endl;
	}
}

void MPIHelper::broadcastMtrys( int rank, std::vector<GridParams> & gridParams ) {
	size_t idx = 0;
	size_t vecLen = gridParams.size();
	std::vector<uint32_t> collectedMtrys( vecLen );
	// Copy mtrys into the collectedMtrys vector (only on rank 0)
	if (rank == 0)
		std::for_each( collectedMtrys.begin(), collectedMtrys.end(), [&](uint32_t &val) { val = gridParams[idx++].mtry; } );

	// Broadcast collectedMtrys to worker processes
	MPI_Bcast( collectedMtrys.data(), vecLen, MPI_UNSIGNED, 0, MPI_COMM_WORLD );
	MPI_Barrier( MPI_COMM_WORLD );

	// Copy collectedMtrys values into the gridParams vector (only worker ranks)
	idx = 0;
	if (rank > 0)
		std::for_each( gridParams.begin(), gridParams.end(), [&](GridParams &val) { val.mtry = collectedMtrys[idx++]; } );

}

// Not sure if still usefull...
void MPIHelper::saveToFileMPI( const double * const cl1, const double * const cl2, const uint32_t nn,
		const std::vector<uint32_t> * const labels, std::string outFilename ) {

	// Check if file already exists. If so, delete the old one
	std::ofstream chkFile( outFilename.c_str(), std::ios::out );
	if (chkFile.good()) {
		std::cout << "Overwriting old file..." << std::endl;
		chkFile.close();
		remove(outFilename.c_str());
	}

	MPI_File	outFile;
	MPI_File_open( MPI_COMM_SELF, outFilename.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_UNIQUE_OPEN, MPI_INFO_NULL, &outFile );

	std::string tempStr;
	std::stringstream ss;
	char newLine = '\n';

	std::for_each( cl1, cl1 + nn, [&outFile, &tempStr, &ss]( double nnn ) {
		ss.clear();
		ss << nnn;
		ss >> tempStr;
		tempStr.push_back(' ');
		MPI_File_write( outFile, tempStr.c_str(), tempStr.size(), MPI_CHAR, MPI_STATUS_IGNORE );
	} );
	MPI_File_write( outFile, &newLine, 1, MPI_CHAR, MPI_STATUS_IGNORE );

	std::for_each( cl2, cl2 + nn, [&outFile, &tempStr, &ss]( double nnn ) {
		ss.clear();
		ss << nnn;
		ss >> tempStr;
		tempStr.push_back(' ');
		MPI_File_write( outFile, tempStr.c_str(), tempStr.size(), MPI_CHAR, MPI_STATUS_IGNORE );
	} );
	MPI_File_write( outFile, &newLine, 1, MPI_CHAR, MPI_STATUS_IGNORE );

	if (!labels->empty()) {
		for_each( labels->begin(), labels->end(), [&outFile, &tempStr, &ss]( uint32_t nnn ) {
			ss.clear();
			ss << nnn;
			ss >> tempStr;
			tempStr.push_back(' ');
			MPI_File_write( outFile, tempStr.c_str(), tempStr.size(), MPI_CHAR, MPI_STATUS_IGNORE );
		} );
		MPI_File_write( outFile, &newLine, 1, MPI_CHAR, MPI_STATUS_IGNORE );
	}
	MPI_File_close( &outFile );
}

void MPIHelper::verbose() {
	std::cout << "  *** MPI Helper class *** " << std::endl;
	std::cout << "MPI world size (total number of MPI processes): " << worldSize << std::endl;
	std::cout << "Distributed folds: " << distributedFolds << std::endl;
	std::cout << "Number of master processes: " << numberOfMasterProcs << std::endl;
	std::cout << "Number of working processes: " << numberOfWorkingProcs << std::endl;
	std::cout << " --- " << std::endl;
	std::cout << "Number of chunks: " << numberOfChunks << std::endl;
	std::cout << "Number of partitions assigned to each chunk: ";
	std::for_each( numberOfPartsPerChunk.begin(), numberOfPartsPerChunk.end(), [](uint32_t &nn) { std::cout << nn << " "; } ); std::cout << std::endl;
	std::cout << "Number of threads in each MPI process: " << numberOfThreadsPerProc << std::endl;
}

MPIHelper::~MPIHelper() {
}
