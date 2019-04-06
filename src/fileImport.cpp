// parSMURF
// Alessandro Petrini, 2018-2019
#include "fileImport.h"

void Importer::import( const ArgHandle * const argHan, std::vector<double>& x, std::vector<uint32_t>& y, std::vector<uint32_t>& f, uint32_t * const nFolds ) {

	uint32_t n;
	uint32_t con = 0;

	// check if datafile is binary or plain text, then open accordingly
	// No checks on the filename or its length. To be added later, in case...
	bool isBinary;
	std::string dataFileExt = argHan->dataFilename.substr( argHan->dataFilename.length() - 4, 4 );
	if ( dataFileExt == ".bin" ) {
		isBinary = true;
		std::cout << "Reading from binary file..." << std::endl;
	} else
		isBinary = false;

	std::ifstream labelFile( argHan->labelFilename.c_str(), std::ios::in );

	y.clear();
	f.clear();
	x.clear();

	if (!labelFile)
		throw std::runtime_error( "Error opening label file." );

	double inDouble;
	uint32_t inUint32;
	std::cout << "Reading label file..." << std::endl;
	while (labelFile >> inUint32) {
		y.push_back( inUint32 );
		con++;
	}
	std::cout << con << " labels read" << std::endl;
	n = con;
	labelFile.close();

	con = 0;
	if (!argHan->foldFilename.empty()) {
		std::ifstream foldFile( argHan->foldFilename.c_str(), std::ios::in );
		if (!foldFile)
			throw std::runtime_error( "Error opening fold file." );

		std::cout << "Reading fold file..." << std::endl;
		*nFolds = 0;
		while (foldFile >> inUint32) {
			f.push_back( inUint32 );
			con++;
			if (f[con - 1] > *nFolds) *nFolds = f[con - 1];
		}
		std::cout << con << " values read." << std::endl;
		(*nFolds)++;
		std::cout << "Total number of folds: " << *nFolds << std::endl;
		foldFile.close();
		if (con != n)
			std::cout << "\033[31;1mWARNING: size mismatch between label and fold file!!!\033[0m" << std::endl;
	}

	con = 0;
	std::cout << "Reading data file..." << std::endl;

	// Import data from tsv or csv file
	if (!isBinary) {
		std::ifstream dataFile( argHan->dataFilename.c_str(), std::ios::in );
		if (!dataFile)
			throw std::runtime_error( "Error opening matrix file." );

		while (dataFile >> inDouble) {
			x.push_back( inDouble );
			con++;
		}
		dataFile.close();
	} else {
		std::ifstream dataFile( argHan->dataFilename.c_str(), std::ios::binary );
		if (!dataFile)
			throw std::runtime_error( "Error opening matrix file." );

		while (dataFile.read( reinterpret_cast<char*>( &inDouble ), sizeof(double) )) {
			x.push_back( inDouble );
			con++;
		}
		dataFile.close();
	}

	std::cout << con << " values read." << std::endl;

	if (con % n != 0)
		std::cout << "\033[31;1mWARNING: size mismatch between label and data file!!!\033[0m" << std::endl;

}

void Importer::test( std::string baseFilename ) {
	std::vector<double> x_tsv( 10 );
	std::vector<double> x_bin( 10 );
	std::string tsvFilename = baseFilename + ".txt";
	std::string binFilename = baseFilename + ".bin";

	double inDouble;
	uint32_t tsv_con = 0;
	uint32_t bin_con = 0;
	Timer tsv_tt, bin_tt;

	// Reading tsv
	std::ifstream tsvFile( tsvFilename.c_str(), std::ios::in );
	if (!tsvFile)
		throw std::runtime_error( "Error opening tsv file." );

	tsv_tt.startTime();
	while (tsvFile >> inDouble) {
		x_tsv.push_back( inDouble );
		tsv_con++;
	}
	tsvFile.close();
	tsv_tt.endTime();
	std::cout << "Import from tsv - time: " << tsv_tt.duration() << std::endl;

	// Reading binary
	std::ifstream binFile( binFilename.c_str(), std::ios::binary );
	if (!binFile)
		throw std::runtime_error( "Error opening bin file." );

	bin_tt.startTime();
	while (binFile.read( reinterpret_cast<char*>( &inDouble ), sizeof(double) )) {
		x_bin.push_back( inDouble );
		bin_con++;
	}
	binFile.close();
	bin_tt.endTime();
	std::cout << "Import from bin - time: " << bin_tt.duration() << std::endl;

	std::cout << "tsv count: " << tsv_con << " - bin count: " << bin_con << std::endl;
	for ( uint32_t ii = 0; ii < tsv_con; ii++ ) {
		if (x_tsv[ii] != x_bin[ii])
			std::cout << "Mismatch at: " << ii << " - tsv: " << x_tsv[ii] << " - bin: " << x_bin[ii] << " - DIFF: " << x_tsv[ii] - x_bin[ii] << std::endl;
	}
}

void Importer::importParameters( std::vector<GridParams> & gridParams ) {
	// As now, parameter filename is hardcoded
	std::ifstream parametersFile( "params.dat", std::ios::in );
	GridParams tempPar;
	gridParams.clear();

	if (!parametersFile) {
		std::cout << "Error opening parameter files" << std::endl;
		std::exit( -1 );
	}

	std::string inStr;
	std::vector<std::string> splittedStr;
	while (std::getline(parametersFile, inStr)) {
		if (inStr.length() == 0)
			continue;
		if (inStr[0] == '#')
			continue;

		splittedStr = split_str( inStr );
		// nRun, nParts, fp, ratio, k, numTrees, mtry <<-- from foldX.dat
		// becomes:
		// nFold, nParts, fp, ratio, k, numTrees, mtry <<-- expected in params.dat
		if (splittedStr.size() != 7) {
			std::cout << "error while reading params.dat" << std::endl;
			continue;
		}

		tempPar.nParts	= atoi( splittedStr[1].c_str() );
		tempPar.fp		= atoi( splittedStr[2].c_str() );
		tempPar.ratio	= atoi( splittedStr[3].c_str() );
		tempPar.k		= atoi( splittedStr[4].c_str() );
		tempPar.nTrees	= atoi( splittedStr[5].c_str() );
		tempPar.mtry	= atoi( splittedStr[6].c_str() );
		gridParams.push_back( tempPar );
	}

	parametersFile.close();
}
