// parSMURF
// Alessandro Petrini, 2018-2019
#include "hyperSMURF.h"

hyperSMURF::hyperSMURF( const CommonParams * const commonParams, std::vector<GridParams> & gridParams,
	Folds * const foldManager, const std::vector<double> & x, const std::vector<uint32_t> & y,
	double * const class1Prob, double * const class2Prob ) :
			commonParams( commonParams ), gridParams( gridParams ), foldManager( foldManager ),
			x( x ), y( y ), class1Prob( class1Prob ), class2Prob( class2Prob ) {
	lockCreated = false;
}

hyperSMURF::~hyperSMURF() {
	if (lockCreated)
		omp_destroy_lock( &accumulLock );
	if (ttd != nullptr)
		delete ttd;
	if (part != nullptr)
		delete part;
}

void hyperSMURF::initStandard() {
	omp_init_lock( &accumulLock );
	lockCreated 	= true;

	nn				= commonParams->nn;
	mm				= commonParams->mm;
	nFolds			= commonParams->nFolds;
	nPart			= gridParams[0].nParts;
	numTrees		= gridParams[0].nTrees;
	fp				= gridParams[0].fp;
	ratio			= gridParams[0].ratio;
	k				= gridParams[0].k;
	mtry			= gridParams[0].mtry;
	seed			= commonParams->seed;
	verboseLevel	= commonParams->verboseLevel;
	nThr			= commonParams->nThr;
	rfThr			= commonParams->rfThr;
	wmode			= commonParams->wmode;
	woptimiz		= commonParams->woptimiz;
	rfVerbose		= commonParams->rfVerbose;
	inInternalCV	= false;
	forestDirname	= commonParams->forestDirname;

	if (verboseLevel > VERBSILENT) std::cout << TXT_BIBLU << "Computing training and test partitions" << TXT_NORML << std::endl;
	ttd = new TestTrainDivider( foldManager, wmode );
	if (verboseLevel == VERBALL) ttd->verbose();

	if (verboseLevel > VERBSILENT) std::cout << TXT_BIBLU << "Computing partitions" << TXT_NORML << std::endl;
	part = new Partition( foldManager, ttd, nPart, fp, ratio, mm, nn );
}

void hyperSMURF::initInternalCV( uint32_t foldToJump ) {
	omp_init_lock( &accumulLock );
	lockCreated = true;

	nn					= commonParams->nn;
	mm					= commonParams->mm;
	nFolds				= commonParams->nFolds - 1;
	seed				= commonParams->seed;
	verboseLevel		= commonParams->verboseLevel;
	nThr				= commonParams->nThr;
	rfThr				= commonParams->rfThr;
	wmode				= MODE_CV;
	woptimiz			= OPT_NO;
	rfVerbose			= commonParams->rfVerbose;
	inInternalCV		= true;

	this->foldToJump	= foldToJump;
	tempcl1Prob			= nullptr;
	tempcl2Prob			= nullptr;

	if (verboseLevel > VERBSILENT) std::cout << TXT_BIPRP << "Computing training and test partitions for internal CV - iteration " << foldToJump << TXT_NORML << std::endl;
	ttd = new TestTrainDivider( foldManager, wmode, foldToJump );
	if (verboseLevel == VERBALL) ttd->verbose();
}

void hyperSMURF::setGridParams( uint32_t idx ) {
	idxInGridParams	= idx;
	nPart			= gridParams[idx].nParts;
	numTrees		= gridParams[idx].nTrees;
	fp				= gridParams[idx].fp;
	ratio			= gridParams[idx].ratio;
	k				= gridParams[idx].k;
	mtry			= gridParams[idx].mtry;
}

void hyperSMURF::createPart() {
	if (verboseLevel > VERBSILENT) std::cout << TXT_BIPRP << "Computing partitions" << TXT_NORML << std::endl;
	part = new Partition(foldManager, ttd, nPart, fp, ratio, mm, nn);
}

bool hyperSMURF::parametersOptimizer( uint32_t foldToJump ) {
	std::string fileName = std::string( "fold" + std::to_string(foldToJump) + ".dat" );
	std::string tempFileName = std::string( "temp" + std::to_string(foldToJump) + ".dat" );
	std::string fileLine;
	std::string commandLine = std::string("python spearmint/spearmint-lite/spearmint-lite.py --method=GPEIOptChooser --grid-size=20000 --method-args=mcmc_iters=10,noiseless=0 --result=" + fileName + " ." + " --config " + commonParams->cfgFilename );
	size_t idxInGrid = 0;
	gridParams.clear();
	hyperSMURF hyper_inner( commonParams, gridParams, foldManager, x, y, tempcl1Prob, tempcl2Prob );
	hyper_inner.initInternalCV( foldToJump );

	for (size_t i = 0; i < 5; i++) {
		int retVal = 0;
		do {
			int retVal = std::system( commandLine.c_str() );
		} while (retVal != 0);

		// Reads the resultfile and look for lines beginning with P (Pending)
		std::ifstream resultFile( fileName.c_str(), std::ios::in );
		std::ofstream tempFile( tempFileName.c_str(), std::ios::out );
		while (std::getline( resultFile, fileLine )) {
			if (fileLine[0] == 'P') {
				GridParams tempGridParam;
				std::vector<std::string> splittedStr = split_str( fileLine, " " );

				std::cout << "Params read: " << splittedStr[2] << " " << splittedStr[3] << " " << splittedStr[4] << " "
					<< splittedStr[5] << " " << splittedStr[6] << " " << splittedStr[7] << std::endl;

				tempGridParam.nParts	= atoi( splittedStr[2].c_str() );
				tempGridParam.fp		= atoi( splittedStr[3].c_str() );
				tempGridParam.ratio		= atoi( splittedStr[4].c_str() );
				tempGridParam.k			= atoi( splittedStr[5].c_str() );
				tempGridParam.nTrees	= atoi( splittedStr[6].c_str() );
				tempGridParam.mtry		= atoi( splittedStr[7].c_str() );
				gridParams.push_back( tempGridParam );

				// Run internal CV with the pending results
				std::fill( tempcl1Prob, tempcl1Prob + nn, 0 );
				std::fill( tempcl2Prob, tempcl2Prob + nn, 0 );
				hyper_inner.setGridParams( idxInGrid );
				hyper_inner.createPart();
				hyper_inner.smurfIt();

				// write the results removing the pending experiment
				tempFile << std::to_string( -(gridParams[i].auprc) ) << " 0 " << splittedStr[2] << " " << splittedStr[3]
					<< " " << splittedStr[4] << " " << splittedStr[5] << " " << splittedStr[6] << " " << splittedStr[7]
					<< std::endl;
				idxInGrid++;

			} else {
				tempFile << fileLine << std::endl;
			}
		}
		resultFile.close();
		tempFile.close();
		std::rename( tempFileName.c_str(), fileName.c_str() );
	}
}

void hyperSMURF::smurfIt() {
	if (woptimiz != OPT_NO) {
		tempcl1Prob = new double[nn];
		tempcl2Prob = new double[nn];
	}
	std::vector<std::string> nomi = generateRandomName( mm + 1 );
	nomi[mm] = std::string( "Labels" );

	uint32_t startingFold = 0;
	uint32_t endingFold = nFolds;
	if ((!inInternalCV) && (commonParams->minFold != -1))
		startingFold = commonParams->minFold;
	if ((!inInternalCV) && (commonParams->maxFold != -1))
		endingFold = commonParams->maxFold;

	// CROSS-VALIDATION AND TRAIN
	if ((wmode == MODE_CV) | (wmode == MODE_TRAIN)) {
		for (uint32_t currentFold = startingFold; currentFold < endingFold; currentFold++) {

			// Inception...
			if (woptimiz != OPT_NO) {
				///// This is the grid exploration section
				if (woptimiz == OPT_GRID) {
					hyperSMURF hyper_inner( commonParams, gridParams, foldManager, x, y, tempcl1Prob, tempcl2Prob );
					hyper_inner.initInternalCV( currentFold );

					std::string statFileName = std::string( "fold" + std::to_string(currentFold) + ".dat" );

					for (uint32_t i = 0; i < gridParams.size(); i++) {
						std::fill( tempcl1Prob, tempcl1Prob + nn, 0 );
						std::fill( tempcl2Prob, tempcl2Prob + nn, 0 );
						hyper_inner.setGridParams( i );
						hyper_inner.createPart();
						hyper_inner.smurfIt();

						std::ofstream tempFile;
						tempFile.open( statFileName.c_str(), std::ios::app );
						tempFile << std::to_string( i ) << " "
							<< std::to_string( gridParams[i].nParts ) << " " << std::to_string( gridParams[i].fp ) << " "
							<< std::to_string( gridParams[i].ratio ) << " " << std::to_string( gridParams[i].k ) << " "
							<< std::to_string( gridParams[i].nTrees ) << " " << std::to_string( gridParams[i].mtry ) << " "
							<< std::to_string( gridParams[i].auroc ) << " " << std::to_string( gridParams[i].auprc ) << " "
							<< std::endl;
						tempFile.close();
					}
				///// Alternatively, explore the parameter hypercube via Bayesian optimization
				} else if (woptimiz == OPT_AUTOGP)
					parametersOptimizer( currentFold );
				/////

				// Finding the maximum auprc for this fold
				auto maxAuprcIdx = std::max_element( gridParams.begin(), gridParams.end(), [](const GridParams i, const GridParams j) {
					return i.auprc < j.auprc;
				} );

				std::cout << "max auprc: " << maxAuprcIdx->auprc << std::endl;
				nPart		= maxAuprcIdx->nParts;
				fp			= maxAuprcIdx->fp;
				ratio		= maxAuprcIdx->ratio;
				k			= maxAuprcIdx->k;
				numTrees	= maxAuprcIdx->nTrees;
				mtry		= maxAuprcIdx->mtry;
				// Also, internals of Partition class must be updated...
				part->nPart = nPart;
				part->fp = fp;
				part->ratio = ratio;
				part->maxSize = ratio == 0 ?
						( (fp + 1) * (foldManager->maxPos) + foldManager->maxNeg ) * (foldManager->nFolds - 1) : // fix per ratio = 0
						(fp + 1) * (ratio + 1) * (foldManager->maxPos) * (foldManager->nFolds - 1);
				// Debug print
				std::cout << TXT_BIGRN << "Setting values of best AUPRC: nPart: " << nPart << " - fp: " << fp << " - ratio: " << ratio <<
					" - k: " << k << " - nTrees: " << numTrees << " - mtry: " << mtry << TXT_NORML << std::endl;
			}

			if (!inInternalCV && (commonParams->customCV)) {
				// Here we set the parameters from the external file, overriding the defaults
				nPart		= gridParams[currentFold].nParts;
				fp			= gridParams[currentFold].fp;
				ratio		= gridParams[currentFold].ratio;
				k			= gridParams[currentFold].k;
				numTrees	= gridParams[currentFold].nTrees;
				mtry		= gridParams[currentFold].mtry;
				// Also, internals of Partition class must be updated...
				part->nPart = nPart;
				part->fp = fp;
				part->ratio = ratio;
				part->maxSize = ratio == 0 ?
						( (fp + 1) * (foldManager->maxPos) + foldManager->maxNeg ) * (foldManager->nFolds - 1) : // fix per ratio = 0
						(fp + 1) * (ratio + 1) * (foldManager->maxPos) * (foldManager->nFolds - 1);
				// Debug print
				std::cout << TXT_BIGRN << "Setting custom values: nPart: " << nPart << " - fp: " << fp << " - ratio: " << ratio <<
					" - k: " << k << " - nTrees: " << numTrees << " - mtry: " << mtry << TXT_NORML << std::endl;
			}

			if (inInternalCV)
				std::cout << TXT_BIPRP;
			else
				std::cout << TXT_BIBLU;
			if (verboseLevel > VERBSILENT) std::cout << "Fold: " << currentFold << TXT_NORML << std::endl;
			part->setFold( currentFold );

			#pragma omp parallel
			{
				std::vector<std::string> nomiPriv = nomi;
				#pragma omp for
				for (int32_t currentPart = 0; currentPart < nPart; currentPart++) {

					#pragma omp critical
					{
						if (inInternalCV)
							std::cout << TXT_BIPRP;
						else
							std::cout << TXT_BIBLU;
						if (verboseLevel > VERBSILENT) std::cout << "   Partition: " << currentPart << " by thread: " << omp_get_thread_num() << std::endl;
						if (verboseLevel > VERBSILENT) std::cout << "      Over/undersampling" << TXT_NORML << std::endl;
					}

					uint32_t seedCustom = seed + currentPart + currentFold * nPart;

					SamplerKNN samp( part, wmode, x.data(), mm, nn, k );
					samp.setPartition( currentPart );
					samp.sample();
					#pragma omp critical
					{
						if (inInternalCV)
							std::cout << TXT_BIPRP;
						else
							std::cout << TXT_BIBLU;
						//if (verboseLevel == VERBALL) samp.verbose();
					}

					nomiPriv[mm] = "Labels";

					std::vector<double> trngDataCopy( samp.lineLen * (mm + 1) );
					transposeMatrix(trngDataCopy.data(), samp.trngData, samp.lineLen, mm + 1);
					std::unique_ptr<Data> input_data( new DataDouble( trngDataCopy, nomiPriv, samp.trngSize, mm + 1 ) );

					// train the random forest...
					#pragma omp critical
					{
						if (inInternalCV)
							std::cout << TXT_BIPRP;
						else
							std::cout << TXT_BIBLU;
						if (verboseLevel > VERBSILENT) std::cout << "      Random forest training" << TXT_NORML << std::endl;
					}
					rfRanger rf( mm, false, std::move(input_data), numTrees, mtry, rfThr, seedCustom );
					rf.train( rfVerbose );

					#pragma omp critical
					{
						if (inInternalCV)
							std::cout << TXT_BIPRP;
						else
							std::cout << TXT_BIBLU;
						if (verboseLevel > VERBPROGR) std::cout << "      Overall prediction error: " << rf.forest->getOverallPredictionError() << TXT_NORML << std::endl;
					}

					// CROSS-VALIDATION ONLY
					if (wmode == MODE_CV) {
						samp.copyTestSet( x.data() );
						nomiPriv[mm] = "dependent";
						std::vector<double> testDataCopy( samp.testSize * (mm + 1) );
						transposeMatrix(testDataCopy.data(), samp.testData, samp.testSize, mm + 1);
						std::unique_ptr<Data> test_data( new DataDouble( testDataCopy, nomiPriv, samp.testSize, mm + 1 ) );

						#pragma omp critical
						{
							if (inInternalCV)
								std::cout << TXT_BIPRP;
							else
								std::cout << TXT_BIBLU;
							if (verboseLevel > VERBSILENT) std::cout << "      Random forest test" << TXT_NORML << std::endl;
						}
						rfRanger rfTest( rf.forest, mm, true, std::move(test_data), numTrees, mtry, rfThr, 2 * seedCustom );
						rfTest.predict( rfVerbose );

						const std::vector<std::vector<std::vector<double>>>& predictions = rfTest.forestPred->getPredictions();

						// ...probability accumulation
						#pragma omp critical
						{
							if (inInternalCV)
								std::cout << TXT_BIPRP;
							else
								std::cout << TXT_BIBLU;
							if (verboseLevel > VERBSILENT) std::cout << "      Accumulating" << TXT_NORML << std::endl;
						}
						omp_set_lock( &accumulLock );
							samp.accumulateAndDivideResInProbVect( predictions, class1Prob, class2Prob, nPart );
						omp_unset_lock( &accumulLock );

					// TRAINING ONLY
					} else if (wmode == MODE_TRAIN) {
						#pragma omp critical
						{
							if (inInternalCV)
								std::cout << TXT_BIPRP;
							else
								std::cout << TXT_BIBLU;
							if (verboseLevel > VERBSILENT) std::cout << "      Saving RF..." << TXT_NORML << std::endl;
						}
						rf.saveForest( currentPart, forestDirname );
					}
				}
			}
			//// Print AUROC and AUPRC for the current fold over the set of best parameters
			// This should be disabled (or optimized) if performance is an issue
			if ((wmode == MODE_CV) | (wmode == MODE_TRAIN)) {

				size_t tempSize = ttd->testNegNum[currentFold] + ttd->testPosNum[currentFold];
				std::vector<uint32_t> tempLabels(tempSize);
				std::vector<double> tempPreds(tempSize);
				size_t tempIdx = 0;
				std::for_each(ttd->testPosIdx[currentFold], ttd->testPosIdx[currentFold] + ttd->testPosNum[currentFold], [&](uint32_t val) mutable {
					tempLabels[tempIdx] = y[val];
					tempPreds[tempIdx++] = class1Prob[val];
				});
				std::for_each(ttd->testNegIdx[currentFold], ttd->testNegIdx[currentFold] + ttd->testNegNum[currentFold], [&](uint32_t val) mutable {
					tempLabels[tempIdx] = y[val];
					tempPreds[tempIdx++] = class1Prob[val];
				});

				// evaluate auroc and auprc
				Curves evalauprc(tempLabels, tempPreds.data());
				// BUG: Do not invert evalAUROC_ok() and evalAUPRC()...
				double auroc = evalauprc.evalAUROC_ok();
				double auprc = evalauprc.evalAUPRC();
				if (inInternalCV)
					std::cout << TXT_BIPRP << "Internal CV on fold " << currentFold << " -> AUROC: "
						<< auroc << " - AUPRC: " << auprc << TXT_NORML << std::endl;
				else {
					if (commonParams->customCV)
						std::cout << TXT_BIBLU << "External CV on fold " << currentFold << " over custom set of parameters -> AUROC: "
							<< auroc << " - AUPRC: " << auprc << TXT_NORML << std::endl;
					else
						std::cout << TXT_BIBLU << "External CV on fold " << currentFold << " over the best parameters -> AUROC: "
							<< auroc << " - AUPRC: " << auprc << TXT_NORML << std::endl;
				}
			}
		}
	}

	// TEST ONLY
	if (wmode == MODE_PREDICT) {
		part->setFold( 0 );
		#pragma omp parallel for
			for (int32_t currentPart = 0; currentPart < nPart; currentPart++) {
				uint32_t seedCustom = seed + currentPart;
				std::vector<std::string> nomiPriv = nomi;
				nomiPriv[mm] = "Labels";

				#pragma omp critical
				{
					if (inInternalCV)
						std::cout << TXT_BIPRP;
					else
						std::cout << TXT_BIBLU;
					if (verboseLevel > VERBSILENT) std::cout << "   Partition: " << currentPart << " by thread: " << omp_get_thread_num() << TXT_NORML << std::endl;
				}

				// setting data
				Sampler samp( part, wmode, x.data(), mm, nn );
				samp.copyTestSet( x.data() );
				nomiPriv[mm] = "dependent";
				std::string forestFilename = forestDirname + "/" + std::to_string( currentPart ) + ".out.forest";
				std::vector<double> testDataCopy( samp.testSize * (mm + 1) );
				transposeMatrix(testDataCopy.data(), samp.testData, samp.testSize, mm + 1);
				std::unique_ptr<Data> test_data( new DataDouble( testDataCopy, nomiPriv, samp.testSize, mm + 1 ) );
				rfRanger rfTest( forestFilename, mm, true, std::move(test_data), numTrees, mtry, rfThr, seedCustom );
				rfTest.predict( rfVerbose );

				const std::vector<std::vector<std::vector<double>>>& predictions = rfTest.forestPred->getPredictions();

				// ...probability accumulation
				#pragma omp critical
				{
					if (inInternalCV)
						std::cout << TXT_BIPRP;
					else
						std::cout << TXT_BIBLU;
					if (verboseLevel > VERBSILENT) std::cout << "      Accumulating" << TXT_NORML << std::endl;
				}
				omp_set_lock( &accumulLock );
					samp.accumulateAndDivideResInProbVect( predictions, class1Prob, class2Prob, nPart );
				omp_unset_lock( &accumulLock );

			}
	}

	// ...and average
	// CROSS-VALIDATION AND TESTING
	if ((wmode == MODE_CV) | (wmode == MODE_PREDICT)) {
		if (inInternalCV)
			std::cout << TXT_BIPRP;
		else
			std::cout << TXT_BIBLU;
		if (verboseLevel > VERBSILENT) std::cout << "Evaluating AUPRC" << TXT_NORML << std::endl;

		if (inInternalCV) {
			size_t	tempSize = 0;
			// Count how many examples
			for (uint32_t i = 0; i < nFolds; i++) {
				tempSize += (ttd->testNegNum[i] + ttd->testPosNum[i]);
			}
			// generate teporary label and prediction vectors
			std::vector<uint32_t> tempLabels(tempSize);
			std::vector<double> tempPreds(tempSize);
			size_t tempIdx = 0;
			for (uint32_t i = 0; i < nFolds; i++) {
				std::for_each(ttd->testPosIdx[i], ttd->testPosIdx[i] + ttd->testPosNum[i], [&](uint32_t val) mutable {
					tempLabels[tempIdx] = y[val];
					tempPreds[tempIdx++] = class1Prob[val];
				});
				std::for_each(ttd->testNegIdx[i], ttd->testNegIdx[i] + ttd->testNegNum[i], [&](uint32_t val) mutable {
					tempLabels[tempIdx] = y[val];
					tempPreds[tempIdx++] = class1Prob[val];
				});
			}

			// evaluate auroc and auprc
			Curves evalauprc(tempLabels, tempPreds.data());
			gridParams[idxInGridParams].auroc = evalauprc.evalAUROC_ok();
			gridParams[idxInGridParams].auprc = evalauprc.evalAUPRC();
			std::cout << "AUROC: " << gridParams[idxInGridParams].auroc << " - AUPRC: " << gridParams[idxInGridParams].auprc
				<< TXT_NORML << std::endl;
		} else {
			Curves evalauprc(y, class1Prob);
			// BUG: Do not invert evalAUROC_ok() and evalAUPRC()...
			double auroc = evalauprc.evalAUROC_ok();
			double auprc = evalauprc.evalAUPRC();
			std::cout << TXT_BIBLU << "AUROC: " << auroc << " - AUPRC: " << auprc << TXT_NORML << std::endl;
		}
	}

	if (woptimiz != OPT_NO) {
		delete[] tempcl2Prob;
		delete[] tempcl1Prob;
	}
}
