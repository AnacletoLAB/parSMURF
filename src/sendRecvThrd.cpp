// parSMURF
// Alessandro Petrini, 2018-2019
#include "sendRecvThrd.h"

std::mutex g_partRecvLock;
std::mutex g_partSendLock;

void recvThrd( uint32_t totThrd, uint32_t currentThrd, uint32_t rank_, uint32_t partitionForThisProc, uint32_t trngDataSize,
	uint32_t m, uint32_t n, uint32_t fp, uint32_t k, uint32_t numTrees, uint32_t mtry, uint32_t rfThr, uint32_t seed,
 	uint32_t testSize, double * const testData, double * tempProb1, double * tempProb2, std::mutex * p_accumulLock, bool MPIverbose,
 	uint8_t wmode, std::string forestDirname ) {

	double	*	trngData = new double[trngDataSize];	checkPtr<double>( trngData, __FILE__, __LINE__ );

	for (uint32_t i = currentThrd; i < partitionForThisProc; i += totThrd ) {
		int			trngSz;

		g_partRecvLock.lock();
			MPI_Status stat2;
			//MPI_Probe( 0, 0, MPI_COMM_WORLD, &stat );
			//MPI_Get_count( &stat, MPI_DOUBLE, &trngSize );
			MPI_Recv( trngData, trngDataSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &stat2 );
			MPI_Get_count( &stat2, MPI_DOUBLE, &trngSz );
			if (MPIverbose) {
				std::cout << "\033[35;1m   rank: " << rank_ << " thread: " << currentThrd << " received training data for part: " << (uint32_t) trngData[1] <<". Sending ack to master proc." << TXT_NORML << std::endl;
			}
			//std::cout << "RANK " << rank_ << " THRD " << currentThrd << " received part: " << trngData[1] << " having actual size of " << trngData[2] << std::endl;
			MPI_Send( &trngSz, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD );
			//std::cout << "RANK " << rank_ << " THRD " << currentThrd << " sending ack." << std::endl;
		g_partRecvLock.unlock();

		// Now, do as per classic hyperSMURF...
		// reading the header:
		uint32_t currentFold		= (uint32_t) trngData[0];
		uint32_t currentPart		= (uint32_t) trngData[1];
		uint32_t trngSize			= (uint32_t) trngData[2];
		uint32_t posForOverSmpl		= (uint32_t) trngData[3];
		uint32_t posToBeGenerated 	= (uint32_t) trngData[4];

		uint32_t seedCustom = seed + currentPart + currentFold * partitionForThisProc;

		double	*	trngData_ = trngData + TRNGHEADERSIZE;

		SamplerKNNMPI samp( trngData_, m, n, trngSize, k, fp, posToBeGenerated, posForOverSmpl );
		samp.sample();

		std::vector<std::string> nomi = generateRandomName( m + 1 );
		nomi[m] = "Labels";

		std::vector<double> trngDataCopy( trngSize * (m + 1) );
		// memcpy(trngDataCopy.data(), trngData_, trngSize * (m + 1) * sizeof(double) );
		transposeMatrix(trngDataCopy.data(), trngData_, trngSize, m + 1);
		std::unique_ptr<Data> input_data( new DataDouble( trngDataCopy, nomi, trngSize, m + 1 ) );
		rfRanger rf( m, false, std::move(input_data), numTrees, mtry, rfThr, seedCustom );
		rf.train( false );

		///A1/// if CV
		if (wmode == MODE_CV) {
			nomi[m] = "dependent";
			std::vector<double> testDataCopy( testSize * (m + 1) );
			// memcpy(testDataCopy.data(), testData, testSize * (m + 1) * sizeof(double) );
			transposeMatrix(testDataCopy.data(), testData, testSize, m + 1);
			std::unique_ptr<Data> test_data( new DataDouble( testDataCopy, nomi, testSize, m + 1 ) );
			rfRanger rfTest( rf.forest, m, true, std::move(test_data), numTrees, mtry, rfThr, seedCustom );
			rfTest.predict( false );

			const std::vector<std::vector<std::vector<double>>>& predictions = rfTest.forestPred->getPredictions();

			p_accumulLock->lock();
				samp.accumulateTempProb( predictions, testSize, tempProb1, tempProb2 );
			p_accumulLock->unlock();

		///A1/// else if TRAIN
		} else if (wmode == MODE_TRAIN) {
			rf.saveForest( currentPart, forestDirname );
		}
	}

	delete[] trngData;

}

void sendThrd( uint32_t totThrd, uint32_t currentThrd, uint32_t mm, uint32_t trngDataSize,
const MPIHelper * const MpiH, std::vector<PartitionMPI> &partMPI, const std::vector<double> & xx, bool MPIverbose ) {

	double	*	trngData = new double[trngDataSize];		checkPtr<double>( trngData, __FILE__, __LINE__ );
	MPI_Status ackStatus;
	uint32_t ack = 0;

	for( uint32_t i = 0; i < MpiH->numberOfPartsPerChunk[currentThrd]; i++ ) {
		uint32_t currentPart = MpiH->startingPartition[currentThrd] + i;
		partMPI[currentPart].assignedToChunk = currentThrd;

		// Assembling the header
		trngData[0] = (double) partMPI[currentPart].currentFold;
		trngData[1] = (double) currentPart;
		trngData[2] = (double) partMPI[currentPart].trngSize;
		trngData[3] = (double) partMPI[currentPart].posForOverSmpl;
		trngData[4] = (double) partMPI[currentPart].posToBeGenerated;

		// Filling the testData and trngData matrices
		trngData += TRNGHEADERSIZE;
		partMPI[currentPart].fillTrngData( xx.data(), trngData );
		trngData -= TRNGHEADERSIZE;

		g_partSendLock.lock();
			if (MPIverbose) {
				std::cout << "\033[35;1mSend trng data of part " << trngData[1] << " to process: " << partMPI[currentPart].assignedToChunk + 1 << " - size: " << trngDataSize << ".\033[0m" << std::endl;
			}
			MPI_Send( trngData, /*partMPI[currentPart].trngSize * (mm + 1) + TRNGHEADERSIZE*/trngDataSize, MPI_DOUBLE,
				partMPI[currentPart].assignedToChunk + 1, 0, MPI_COMM_WORLD );
			MPI_Recv( &ack, 1, MPI_UNSIGNED, partMPI[currentPart].assignedToChunk + 1, MPI_ANY_TAG, MPI_COMM_WORLD, &ackStatus );
			if (MPIverbose) {
				std::cout << "\033[35;1mMaster proc received ack for part " << trngData[1] << " from process: " << partMPI[currentPart].assignedToChunk + 1 << ".\033[0m" << std::endl;
			}
		g_partSendLock.unlock();
	}
	delete[] trngData;
}

void predictRecvThrd( uint32_t totThrd, uint32_t currentThrd, uint32_t rank_, uint32_t partitionForThisProc,
	uint32_t m, uint32_t numTrees, uint32_t mtry, uint32_t rfThr, uint32_t seed,
	uint32_t testSize, double * const testData, double * tempProb1, double * tempProb2, std::mutex * p_accumulLock,
	bool MPIverbose, std::string forestDirname ) {

	for ( uint32_t i = currentThrd; i < partitionForThisProc; i += totThrd ) {
		std::vector<std::string> nomi = generateRandomName( m + 1 );
		nomi[m] = "dependent";
		uint32_t seedCustom = seed + i;
		uint32_t currentPart;

		g_partRecvLock.lock();
			MPI_Status stat2;
			MPI_Recv( &currentPart, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, &stat2 );
			if (MPIverbose) {
				std::cout << "\033[35;1m   rank: " << rank_ << " thread: " << currentThrd << " received training partition number: " << currentPart <<". Sending ack to master proc.\033[0m" << std::endl;
			}
			MPI_Send( &currentPart, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD );
		g_partRecvLock.unlock();

		std::string forestFilename = forestDirname + "/" + std::to_string( currentPart ) + ".out.forest";

		std::vector<double> testDataCopy( testSize * (m + 1) );
		// memcpy(testDataCopy.data(), testData, testSize * (m + 1) * sizeof(double) );
		transposeMatrix(testDataCopy.data(), testData, testSize, m + 1);
		std::unique_ptr<Data> test_data( new DataDouble( testDataCopy, nomi, testSize, m + 1 ) );
		rfRanger rfTest( forestFilename, m, true, std::move(test_data), numTrees, mtry, rfThr, seedCustom );
		rfTest.predict( false );

		const std::vector<std::vector<std::vector<double>>>& predictions = rfTest.forestPred->getPredictions();

		p_accumulLock->lock();
			//samp.accumulateTempProb( predictions, testSize, tempProb1, tempProb2 );
			// From samplerKNNMPI.cpp
			for (uint32_t i = 0; i < testSize; i++) {
				tempProb1[i] += predictions[0][i][0];
				tempProb2[i] += predictions[0][i][1];
			}
		p_accumulLock->unlock();

	}

}
